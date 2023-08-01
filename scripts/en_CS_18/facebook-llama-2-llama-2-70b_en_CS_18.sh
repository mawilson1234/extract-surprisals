#!/bin/bash

#SBATCH --job-name=facebook-llama-2-llama-2-70b-en_CS_18
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=288G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_CS_18/facebook-llama-2-llama-2-70b_en_CS_18.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'facebook/llama-2/llama-2-70b' \
	--test_file data/en_CS_18/en_CS_18.txt.gz \
	--tokenizer_name facebook/llama-2/tokenizer.model \
	--save_tmp