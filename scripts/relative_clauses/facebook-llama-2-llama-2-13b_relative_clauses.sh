#!/bin/bash

#SBATCH --job-name=facebook-llama-2-llama-2-13b-relative_clauses
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=78G
#SBATCH --partition=day
#SBATCH --time=16:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/relative_clauses/facebook-llama-2-llama-2-13b_relative_clauses.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'facebook/llama-2/llama-2-13b' \
	--test_file data/relative_clauses/relative_clauses.txt.gz \
	--tokenizer_name facebook/llama-2/tokenizer.model