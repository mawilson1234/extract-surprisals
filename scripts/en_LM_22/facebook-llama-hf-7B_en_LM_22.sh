#!/bin/bash

#SBATCH --job-name=facebook-llama-hf-7B-en_LM_22
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_LM_22/facebook-llama-hf-7B_en_LM_22.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'facebook/llama-hf/7B' \
	--test_file data/en_LM_22/en_LM_22.txt.gz