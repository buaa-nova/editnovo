#!/usr/bin/env bash
set -euo pipefail

# ————— CONFIG —————
DATA_DIR=/root/data/dda-train-data/nine-species-main-mgf-sample/
EXCLUDE_SPECIES="H.-sapiens"            # basename without .mgf
TEST_DIR=/root/data/dda-train-data/nine-species-main-test-sample       # directory with your test .mgf files
PYTHON_MODULE="casanovo.casanovo"
# ————————————

# 1) Gather all .mgf files in DATA_DIR except the excluded species
train_files=()
while IFS= read -r -d '' mgf; do
  name=$(basename "$mgf" .mgf)
  if [[ "$name" != "$EXCLUDE_SPECIES" ]]; then
    train_files+=("$mgf")
  fi
done < <(find "$DATA_DIR" -maxdepth 1 -type f -name '*.mgf' -print0)
# MODEL_PATH="/root/attennovo/checkpoint/epoch=48-step=80212.ckpt"
# 2) Build the base command
cmd=( python -m "$PYTHON_MODULE" train "${train_files[@]}" )

# 3) Append each test file with a '-p' flag
while IFS= read -r -d '' testmgf; do
  cmd+=(-p "$testmgf")
done < <(find "$TEST_DIR" -maxdepth 1 -type f -name '*.mgf' -print0)

# 4) Echo (and run) the command
echo "Running:" "${cmd[@]}"
# exec "${cmd[@]}"
