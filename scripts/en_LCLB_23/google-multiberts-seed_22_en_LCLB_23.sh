#!/bin/bash

#SBATCH --job-name=google-multiberts-seed_22-en_LCLB_23
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_LCLB_23/google-multiberts-seed_22_en_LCLB_23.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'google/multiberts-seed_22' \
	--test_file data/en_LCLB_23/en_LCLB_23.txt.gz