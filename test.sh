#!/bin/bash 

# Directory containing MGF files
data_dir="/root/9speciesbenchmark"

# List of all MGF directories in the main directory
mgf_files=(
    "Apis-mellifera" "Candidatus-endoloripes" "Methanosarcina-mazei" "Saccharomyces-cerevisiae" "Vigna-mungo"
    "Bacillus-subtilis" "H.-sapiens" "Mus-musculus" "Solanum-lycopersicum"
)

# Specify species to exclude
exclude_species="H.-sapiens"  # Change this to exclude a different species

# Filter MGF directories to exclude the specified species
train_files=()
file_count=0
for dir in "${mgf_files[@]}"; do
    if [[ "$dir" != *"$exclude_species"* ]]; then
        for file in "$data_dir/$dir"/*; do
            if [[ -f "$file" ]]; then
                train_files+=("$file")  # Expand wildcards before storing
                ((file_count++))
            fi
        done
    fi
done

# Print the number of training files
echo "Total training files: $file_count"