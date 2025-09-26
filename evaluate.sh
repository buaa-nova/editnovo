#!/usr/bin/env bash
set -euo pipefail

# Directory containing your .mgf files
# DATA_DIR="/root/data/dda-train-data/revised-nine-species/H.-sapiens-sample"
# DATA_DIR="/root/data/dda-train-data/nine-species-main-test-sample"
mgf_file="/root/data/massivekb/ab2797af023147a9943ec09f420f4e84.hdf5"
# Checkpoint to use
CKPT="/root/attennovo/checkpoint/epoch=10-step=294000.ckpt"

# Collect all .mgf files (will be an array of absolute paths)
# mgf_files=("$DATA_DIR"/*.mgf)

# # Verify we found some files
# if [ ${#mgf_files[@]} -eq 0 ]; then
#   echo "Error: no .mgf files found in $DATA_DIR" >&2
#   exit 1
# fi  


# Run evaluation with each file explicitly listed
# python -m casanovo.casanovo evaluate "${mgf_files[@]}" -m "$CKPT"
python -m casanovo.casanovo evaluate "$mgf_file" -m "$CKPT"
