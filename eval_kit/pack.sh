#!/usr/bin/env bash
# 将本评测包打成可传输的 tar.gz（不含结果/缓存）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}/dist}"
STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="ad_gen_eval_kit_${STAMP}"
STAGE="${OUT_DIR}/${NAME}"

mkdir -p "${STAGE}"
rsync -a \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'dist/' \
  --exclude 'eval_results/' \
  --exclude 'ckpt_*/' \
  --exclude '.git/' \
  "${SCRIPT_DIR}/" "${STAGE}/"

mkdir -p "${OUT_DIR}"
TAR="${OUT_DIR}/${NAME}.tar.gz"
tar -czf "${TAR}" -C "${OUT_DIR}" "${NAME}"
rm -rf "${STAGE}"

echo "已打包: ${TAR}"
echo "传输后解压，进入目录执行:"
echo "  pip install -r requirements.txt"
echo "  python evaluate.py --help"
