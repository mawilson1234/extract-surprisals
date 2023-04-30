#!/bin/bash

#SBATCH --job-name=facebook-llama-30B-agreement
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=169G
#SBATCH --partition=bigmem
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/agreement/facebook-llama-30B_agreement.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'facebook/llama/30B' \
	--test_file data/agreement/agreement.txt.gz \
	--tokenizer_name facebook/llama/tokenizer.model