#!/bin/bash
# Download CodeQA source data from HuggingFace for VAGEN training.
#
# Downloads 4 specific files (not snapshot) to avoid HF API rate limits:
#   1. prepared_dataset/all_repos_dataset.jsonl  (315MB)
#   2. contaminated_questions_v3.json            (7KB)
#   3. nontruncated_repos.json                   (10KB)
#   4. code_images_8pt.tar.gz                    (139MB) → extracted locally
#
# Usage:
#   cd /root/projects/viewsuite/VAGEN-codeqa
#   conda activate viewsuite
#   bash examples/train/codeqa/download_data.sh

set -euo pipefail

BASEDIR="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="${BASEDIR}/data/codeqa"
HF_REPO="spatel-learn/codeqa-vagen-trajectories"

echo "============================================"
echo "CodeQA Data Download (4 files only)"
echo "HF Repo:  ${HF_REPO}"
echo "Data Dir: ${DATA_DIR}"
echo "============================================"

mkdir -p "${DATA_DIR}"

# ── Download 4 individual files (avoids rate limit) ──
python3 << PYEOF
from huggingface_hub import hf_hub_download
import os

data_dir = "${DATA_DIR}"
repo = "${HF_REPO}"

files = [
    ("source_data/prepared_dataset/all_repos_dataset.jsonl", "all_repos_dataset.jsonl"),
    ("source_data/contaminated_questions_v3.json",           "contaminated_questions_v3.json"),
    ("source_data/nontruncated_repos.json",                  "nontruncated_repos.json"),
    ("source_data/code_images_8pt.tar.gz",                   "code_images_8pt.tar.gz"),
]

for hf_path, local_name in files:
    dst = os.path.join(data_dir, local_name)
    if os.path.exists(dst):
        print(f"  Skip (exists): {local_name}")
        continue
    print(f"  Downloading: {hf_path} ...")
    downloaded = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=hf_path,
        local_dir=data_dir,
    )
    # hf_hub_download preserves the HF path structure; move to flat layout
    src = os.path.join(data_dir, hf_path)
    if os.path.exists(src) and src != dst:
        os.rename(src, dst)
        print(f"    -> {local_name}")

# Clean up empty HF subdirectories
import shutil
for d in ["source_data/prepared_dataset", "source_data"]:
    p = os.path.join(data_dir, d)
    if os.path.isdir(p) and not os.listdir(p):
        os.rmdir(p)

print("Download complete!")
PYEOF

# ── Extract images ──
if [[ -f "${DATA_DIR}/code_images_8pt.tar.gz" ]] && [[ ! -d "${DATA_DIR}/code_images_8pt" ]]; then
    echo ""
    echo "Extracting code_images_8pt.tar.gz..."
    tar -xzf "${DATA_DIR}/code_images_8pt.tar.gz" -C "${DATA_DIR}"
    echo "Done."
fi

# ── Verify ──
echo ""
echo "Verifying..."
ALL_OK=true

for f in all_repos_dataset.jsonl contaminated_questions_v3.json nontruncated_repos.json; do
    if [[ -f "${DATA_DIR}/${f}" ]]; then
        echo "  OK  ${f}  ($(du -sh "${DATA_DIR}/${f}" | cut -f1))"
    else
        echo "  MISSING  ${f}"
        ALL_OK=false
    fi
done

if [[ -d "${DATA_DIR}/code_images_8pt" ]]; then
    N=$(ls -d "${DATA_DIR}/code_images_8pt"/*/ 2>/dev/null | wc -l)
    echo "  OK  code_images_8pt/  (${N} repos)"
else
    echo "  MISSING  code_images_8pt/"
    ALL_OK=false
fi

echo ""
if $ALL_OK; then
    echo "All data ready at: ${DATA_DIR}"
else
    echo "ERROR: Some files missing."
    exit 1
fi
