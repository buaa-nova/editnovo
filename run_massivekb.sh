# python -m casanovo.casanovo train /root/data/massivekb/massivekb_82c0124b_val.mgf -p /root/data/massivekb/massivekb_82c0124b_val.mgf
#!/usr/bin/env bash
set -euo pipefail



# 3) 构造训练命令
train_command=(
#   python -m casanovo.casanovo train /root/data/massivekb/casanovo/massivekb_82c0124b_train.mgf
python -m casanovo.casanovo train /dev/shm/massivekb/369c9972a4984395b7c7e89bee00f217.hdf5 -p /dev/shm/massivekb/ab2797af023147a9943ec09f420f4e84.hdf5
)

# # 4) 加入测试集（-p 前缀）
# test_dir="/root/data/dda-train-data/revised-nine-species/Bacillus-subtilis-sample/"
# for mgf in "$test_dir"/*.[mM][gG][fF]; do
#   [[ -f "$mgf" ]] || continue
#   train_command+=(-p "$mgf")
# done
# # # 4) 加上 -m model_path
# model_path="/root/attennovo/checkpoint/epoch=6-step=38290.ckpt"
# train_command+=( -m "$model_path" )

# 5) 打印并执行
echo "Training command:"
printf "  %q\n" "${train_command[@]}"
"${train_command[@]}"