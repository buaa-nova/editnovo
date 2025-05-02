#!/bin/bash 

# List of all MGF directories in the main directory
mgf_files=(
    "Apis-mellifera" "Candidatus-endoloripes" "Methanosarcina-mazei" "Saccharomyces-cerevisiae" "Vigna-mungo"
    "Bacillus-subtilis" "H.-sapiens" "Mus-musculus" "Solanum-lycopersicum"
)


# Directory containing MGF files
data_dir="/root/9speciesbenchmark"

# # List of all MGF directories in the main directory
# mgf_files=(
#     "Apis-mellifera" "Candidatus-endoloripes" 
# )

# Specify species to exclude
exclude_species="H.-sapiens"  # Change this to exclude a different species

# Filter MGF directories to exclude the specified species
train_files=()
for dir in "${mgf_files[@]}"; do
    if [[ "$dir" != *"$exclude_species"* ]]; then
        for file in "$data_dir/$dir"/*; do
            if [[ -f "$file" ]]; then
                train_files+=("$file")  # Expand wildcards before storing
            fi
        done
    fi
done

# Construct the training command
train_command=(
    "python" "-m" "casanovo.casanovo" "train"
    "${train_files[@]}"
)

# Add validation files with -p before each file
for file in /root/training_data/test/*; do
    if [[ -f "$file" ]]; then
        train_command+=("-p" "$file")
    fi
done

# Print the command for verification
echo "Training command: ${train_command[*]}"

# Execute the training command
"${train_command[@]}"
