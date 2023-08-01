#!/bin/bash

#SBATCH --job-name=facebook-llama-2-llama-2-13b-en_LM_22
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=78G
#SBATCH --partition=day
#SBATCH --time=16:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_LM_22/facebook-llama-2-llama-2-13b_en_LM_22.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'facebook/llama-2/llama-2-13b' \
	--test_file data/en_LM_22/en_LM_22.txt.gz \
	--tokenizer_name facebook/llama-2/tokenizer.model