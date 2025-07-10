#!/usr/bin/env bash
set -euo pipefail

# 1) 根目录和排除物种
data_dir="/root/data/dda-train-data/nine-species-main-mgf-sample"
exclude_species="H.-sapiens"

# 2) 收集训练文件
train_files=()
for species_path in "$data_dir"/*; do
  [[ -d "$species_path" ]] || continue
  species=$(basename "$species_path")
  if [[ "$species" == "$exclude_species" ]]; then
    echo "Skipping excluded species: $species"
    continue
  fi
  # 遍历该物种目录下的所有 .mgf
  for mgf in "$species_path"/*.[mM][gG][fF]; do
    [[ -f "$mgf" ]] || continue
    train_files+=("$mgf")
  done
done

# 3) 构造训练命令
train_command=(
  python -m casanovo.casanovo train
  "${train_files[@]}"
)

# 4) 加入测试集（-p 前缀）
test_dir="/root/data/dda-train-data/nine-species-main-test-sample"
for mgf in "$test_dir"/*.[mM][gG][fF]; do
  [[ -f "$mgf" ]] || continue
  train_command+=(-p "$mgf")
done
# # 4) 加上 -m model_path
# model_path="/root/data/attennovo/epoch=4-step=27350-v1.ckpt"
# train_command+=( -m "$model_path" )

# 5) 打印并执行
echo "Training command:"
printf "  %q\n" "${train_command[@]}"
# "${train_command[@]}"