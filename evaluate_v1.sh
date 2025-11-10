#!/usr/bin/env bash
set -euo pipefail
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  # Use GPU 2
# Directory containing your .mgf files
# mgf_file="/root/data/dda-train-data/nine-species-v1/Casanovo_preprocessed_data/preproc.high.human.PXD004424.mgf"
mgf_file="/root/data/dda-train-data/nine-species-v1/Casanovo_preprocessed_data/preproc.high.ricebean.PXD005025.mgf"
# mgf_file="/root/attennovo/output_E_Coli.with_seq.mgf"
# DATA_DIR="/root/data/dda-train-data/revised_nine_species/9speciesbenchmark/Solanum-lycopersicum"
# # DATA_DIR="/root/data/dda-train-data/nine-species-v1/Casanovo_preprocessed_data/preproc.high.bacillus.PXD004565.mgf"
# # DATA_DIR="/root/data/dda-train-data/nine-species-main-test-sample"
python -m casanovo.casanovo evaluate "$mgf_file" -m "/root/data/attennovo/epoch=10-step=294000-703.ckpt"
# # Checkpoint to use
# CKPT="/root/data/attennovo/epoch=9-step=273000-689.ckpt"

# # Collect all .mgf files (will be an array of absolute paths)
# # mgf_files=("$DATA_DIR"/*.mgf)

# # Verify we found some files
# if [ ${#mgf_files[@]} -eq 0 ]; then
#   echo "Error: no .mgf files found in $DATA_DIR" >&2
#   exit 1
# fi  


# # Run evaluation with each file explicitly listed
# python -m casanovo.casanovo evaluate "${mgf_files[@]}" -m "$CKPT"
