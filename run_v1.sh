#!/usr/bin/env bash
set -euo pipefail

# # 1) MGF 文件目录 & 要排除的 species（这里排除 human）
# data_dir="$HOME/data/dda-train-data/nine-species-v1/Casanovo_preprocessed_data"
# exclude_species="ricebean"

# # 2) 收集训练文件（直接在 data_dir 下遍历所有 .mgf）
# train_files=()
# for mgf in "$data_dir"/*.[mM][gG][fF]; do
#   [[ -f "$mgf" ]] || continue

#   # 从文件名提取 species：preproc.high.<species>.<id>.mgf
#   base=$(basename "$mgf")
#   species=$(echo "$base" | cut -d. -f3)

#   # 排除指定 species
#   if [[ "$species" == "$exclude_species" ]]; then
#     echo "Skipping excluded species: $species"
#     continue
#   fi

#   train_files+=("$mgf")
# done

# 3) 构造训练命令
train_command=(
python -m casanovo.casanovo train /mnt/hdf5/cfbae77b2a5b4d60b93589eeeaa5abac.hdf5 -p /mnt/hdf5/e9c928edc3b84941896752c5ce5b1ad7.hdf5
)
# # 3) 构造训练命令
# train_command=(python -m casanovo.casanovo train "${train_files[@]}")

# # 4) 加入测试集（-p 前缀）
# test_path="/root/data/dda-train-data/nine-species-v1/sample/preproc.high.ricebean.PXD005025.mgf"
# # 
# train_command+=(-p "$test_path")

# # # 5) 加上已训练好的模型权重
model_path="/root/data/attennovo/v1/ricebean/epoch=1-step=29770-v1.ckpt"
train_command+=( -m "$model_path" )

# # 6) 打印并执行
echo "Training command:"
printf "  %q\n" "${train_command[@]}"
"${train_command[@]}"

