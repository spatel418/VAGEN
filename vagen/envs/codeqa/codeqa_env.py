from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from vagen.envs.gym_image_env import GymImageEnv
from vagen.envs.codeqa.utils.data_loader import (
    load_dataset,
    load_contaminated_ids,
    load_nontruncated_info,
    filter_dataset,
    load_repo_images,
    load_repo_images_stratified,
)
from vagen.envs.codeqa.utils.prompt import (
    system_prompt,
    ocr_observation,
    qa_observation,
    IMAGE_PLACEHOLDER,
)
from vagen.envs.codeqa.utils.answer_extraction import extract_answer_letter


@dataclass
class CodeQAEnvConfig:
    """Configuration for the CodeQA environment."""

    # Data paths (absolute paths required)
    dataset_path: str = ""
    images_dir: str = ""
    contaminated_path: str = ""
    nontruncated_path: str = ""

    # Filtering
    subset: str = "clean_nt"  # "all", "clean", "non_truncated", "clean_nt"

    # Repo-level split filtering (train/val/test)
    # Path to split_definition.json; if set with split, only repos in that split are kept
    split_definition_path: str = ""
    split: str = ""  # "train", "val", "test", or "" to disable

    # Vision token budget for image loading
    max_vision_tokens: int = 230000

    # Stratified subsampling (for training data generation)
    # Set token_budget > 0 to enable stratified subsampling (~20K recommended)
    # Set to 0 to disable (uses max_vision_tokens instead)
    token_budget: int = 0

    # Trajectory ID (informational, not used for subsampling)
    trajectory_id: int = 0

    # Reward
    correct_reward: float = 1.0

    # Image placeholder token
    image_placeholder: str = "<image>"


class CodeQA(GymImageEnv):
    """
    Two-turn CodeQA environment for evaluating VLM code understanding.

    Turn 1 (OCR): Show code images -> model transcribes code
    Turn 2 (QA):  Show MCQ question -> model answers A/B/C/D -> score

    Extends GymImageEnv with the standard async API:
      system_prompt() -> Dict
      reset(seed) -> (obs, info)
      step(action_str) -> (obs, reward, done, info)
      close() -> None
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.cfg = CodeQAEnvConfig(**env_config)

        # Load and filter dataset (cached at module level)
        full_data = load_dataset(self.cfg.dataset_path)
        contaminated_ids = load_contaminated_ids(self.cfg.contaminated_path)
        nt_info = load_nontruncated_info(self.cfg.nontruncated_path)
        nt_sample_ids = set(nt_info.get("sample_ids", []))

        self.samples = filter_dataset(
            full_data, self.cfg.subset, contaminated_ids, nt_sample_ids
        )

        # Apply repo-level split filtering if configured
        if self.cfg.split and self.cfg.split_definition_path:
            with open(self.cfg.split_definition_path, "r", encoding="utf-8") as f:
                split_def = json.load(f)
            if self.cfg.split not in split_def:
                raise ValueError(
                    f"Unknown split '{self.cfg.split}'. "
                    f"Available: {[k for k in split_def if k not in ('description', 'seed', 'token_budget')]}"
                )
            allowed_repos = set(split_def[self.cfg.split])
            self.samples = [s for s in self.samples if s["repo"] in allowed_repos]

        if not self.samples:
            raise ValueError(
                f"No samples after filtering with subset='{self.cfg.subset}'. "
                f"Check data paths and filter criteria."
            )

        # Episode state
        self._current_sample: Optional[Dict[str, Any]] = None
        self._current_images: List[Image.Image] = []
        self._turn: int = 0
        self._ocr_response: str = ""

    async def system_prompt(self) -> Dict[str, Any]:
        """Return system prompt (text only, no images)."""
        return {"obs_str": system_prompt()}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset: select a sample based on seed, load images, return Turn 1 obs.

        Seed-to-sample mapping: index = seed % len(samples)
        """
        index = seed % len(self.samples)
        self._current_sample = self.samples[index]
        self._turn = 1
        self._ocr_response = ""

        # Load images in a thread to avoid blocking the event loop
        repo = self._current_sample["repo"]

        if self.cfg.token_budget > 0:
            # Stratified subsampling: use seed only so all trajectories see the same images
            subsample_seed = seed
            self._current_images, total_avail, tokens_used = await asyncio.to_thread(
                load_repo_images_stratified,
                repo,
                self.cfg.images_dir,
                self._current_sample["context"],
                self.cfg.token_budget,
                subsample_seed,
            )
        else:
            self._current_images, total_avail, tokens_used = await asyncio.to_thread(
                load_repo_images,
                repo,
                self.cfg.images_dir,
                self.cfg.max_vision_tokens,
            )

        num_images = len(self._current_images)

        # Build Turn 1 observation (OCR): images + instruction
        obs_str = ocr_observation(num_images)
        obs: Dict[str, Any] = {"obs_str": obs_str}

        if self._current_images:
            obs["multi_modal_input"] = {
                self.cfg.image_placeholder: list(self._current_images)
            }

        info: Dict[str, Any] = {
            "sample_id": self._current_sample["id"],
            "repo": repo,
            "num_images": num_images,
            "total_images_available": total_avail,
            "vision_tokens_used": tokens_used,
            "seed": seed,
            "sample_index": index,
        }

        return obs, info

    async def step(
        self, action_str: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step.

        Turn 1 -> Turn 2: Accept OCR response, present QA question.
        Turn 2 -> Done:   Extract answer, score, return final reward.
        """
        if self._turn == 1:
            # ---- Turn 1 -> Turn 2 transition ----
            self._ocr_response = action_str
            self._turn = 2

            # Build Turn 2 observation (QA): question text only, no images
            question = self._current_sample["question"]
            obs_str = qa_observation(question)
            obs: Dict[str, Any] = {"obs_str": obs_str}

            info: Dict[str, Any] = {
                "turn": 2,
                "ocr_response_length": len(action_str),
            }

            return obs, 0.0, False, info

        elif self._turn == 2:
            # ---- Turn 2 -> Done ----
            predicted = extract_answer_letter(action_str)
            correct = self._current_sample["correct_letter"].upper()

            is_correct = predicted == correct
            is_nic = predicted == "NOT_IN_CONTEXT"
            is_none = predicted == "NONE"

            reward = self.cfg.correct_reward if is_correct else 0.0

            obs: Dict[str, Any] = {"obs_str": "Episode complete."}

            info: Dict[str, Any] = {
                "turn": 2,
                "success": is_correct,
                "predicted_letter": predicted,
                "correct_letter": correct,
                "is_not_in_context": is_nic,
                "is_none": is_none,
                "sample_id": self._current_sample["id"],
                "repo": self._current_sample["repo"],
                "metrics": {
                    "turn_metrics": {
                        "answer_extracted": predicted not in ("NONE", "NOT_IN_CONTEXT"),
                    },
                    "traj_metrics": {
                        "success": is_correct,
                        "is_not_in_context": is_nic,
                        "predicted_letter": predicted,
                        "correct_letter": correct,
                    },
                },
            }

            return obs, reward, True, info

        else:
            raise RuntimeError(f"Unexpected turn state: {self._turn}")

    async def close(self) -> None:
        """Release image memory."""
        self._current_images = []
        self._current_sample = None
