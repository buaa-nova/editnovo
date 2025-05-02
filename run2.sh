#!/bin/bash

# 设置 MGF 文件所在目录
data_dir="/root/data/dda-train-data/nine-species-main-mgf"

# 指定要排除的物种（以 .mgf 文件名匹配）
exclude_species="H.-sapiens"  # 修改这里可排除其他物种

# 初始化训练文件数组
train_files=()

# 遍历所有 .mgf 文件，排除指定的物种
for file in "$data_dir"/*.mgf; do
    if [[ -f "$file" && "$file" != *"$exclude_species"* ]]; then
        train_files+=("$file")  # 添加符合条件的文件
    fi
done

# 构造训练命令
train_command=(
    "python" "-m" "casanovo.casanovo" "train"
    "${train_files[@]}"
)

# 添加测试集（使用 -p 作为前缀）
for file in /root/training_data/test/*.mgf; do
    if [[ -f "$file" ]]; then
        train_command+=("-p" "$file")
    fi
done

# 打印最终训练命令（用于调试）
echo "Training command: ${train_command[*]}"

# 执行训练命令
"${train_command[@]}"
