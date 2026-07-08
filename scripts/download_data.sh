#!/usr/bin/env bash
# MAST-Data 다운로드 (HuggingFace, CC-BY-4.0 — 저작자: MAST authors, arXiv:2503.13657)
# 데이터는 repo에 넣지 않는다 — 출처에서 직접 받는다.
set -euo pipefail

DEST="$(dirname "$0")/../data/mast"
mkdir -p "$DEST"

BASE="https://huggingface.co/datasets/mcemri/MAST-Data/resolve/main"
curl -sL -o "$DEST/MAD_full_dataset.json" "$BASE/MAD_full_dataset.json"
curl -sL -o "$DEST/MAD_human_labelled_dataset.json" "$BASE/MAD_human_labelled_dataset.json"

ls -la "$DEST"
