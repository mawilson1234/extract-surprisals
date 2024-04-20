#!/bin/bash

#SBATCH --job-name=facebook-llama-hf-30B-en_CS_18
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=169G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts\en_CS_18\facebook-llama-hf-30B_en_CS_18.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'facebook/llama-hf/30B' \
	--test_file data/en_CS_18/en_CS_18.txt.gz \
	--save_tmp