#!/bin/bash

#SBATCH --job-name=facebook-llama-hf-13B-en_LM_22
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=78G
#SBATCH --partition=day
#SBATCH --time=16:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_LM_22/facebook-llama-hf-13B_en_LM_22.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'facebook/llama-hf/13B' \
	--test_file data/en_LM_22/en_LM_22.txt.gz \
	--tokenizer_name facebook/llama/tokenizer.model