#!/usr/bin/env bash
set -euo pipefail
# export CUDA_VISIBLE_DEVICES=2
# export CUDA_VISIBLE_DEVICES=0  # Use GPU 2
# Directory containing your .mgf files
# mgf_file="/root/data/dda-train-data/nine-species-v1/Casanovo_preprocessed_data/preproc.high.human.PXD004424.mgf"
# mgf_file="/root/data/dda-train-data/nine-species-v1/Casanovo_preprocessed_data/preproc.high.ricebean.PXD005025.mgf"
# mgf_file="/root/attennovo/output_E_Coli.with_seq.mgf"
DATA_DIR="/root/data/dda-train-data/revised_nine_species/9speciesbenchmark/Solanum-lycopersicum"
# # DATA_DIR="/root/data/dda-train-data/nine-species-v1/editnovo_preprocessed_data/preproc.high.bacillus.PXD004565.mgf"
# # DATA_DIR="/root/data/dda-train-data/nine-species-main-test-sample"
# python -m editnovo.editnovo evaluate "$mgf_file" -m "/root/data/attennovo/epoch=10-step=294000-703.ckpt"
# python -m editnovo.editnovo train ./sample_preprocessed_spectra.mgf -p ./sample_preprocessed_spectra.mgf 
# python -m editnovo.editnovo evaluate ./sample_preprocessed_spectra.mgf -m ./editnovo-massive-kb.ckpt
# CUDA_VISIBLE_DEVICES=0  python -m editnovo.editnovo evaluate ./sample_preprocessed_spectra.mgf -m ./editnovo-massive-kb.ckpt
# # Checkpoint to use
CKPT="/root/data/attennovo/epoch=10-step=294000-703.ckpt"

# # Collect all .mgf files (will be an array of absolute paths)
mgf_files=("$DATA_DIR"/*.mgf)

# Verify we found some files
if [ ${#mgf_files[@]} -eq 0 ]; then
  echo "Error: no .mgf files found in $DATA_DIR" >&2
  exit 1
fi  


# # Run evaluation with each file explicitly listed
python -m editnovo.editnovo evaluate "${mgf_files[@]}" -m "$CKPT"
