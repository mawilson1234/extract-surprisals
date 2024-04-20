#!/bin/bash

#SBATCH --job-name=meta-llama-Llama-2-70b-hf-relative_clauses
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=288G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/relative_clauses/meta-llama-Llama-2-70b-hf_relative_clauses.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'meta-llama/Llama-2-70b-hf' \
	--test_file data/relative_clauses/relative_clauses.txt.gz \
	--save_tmp \
	--token ~/.hf_auth_token