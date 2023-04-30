#!/bin/bash

#SBATCH --job-name=facebook-llama-30B-attachment_ambiguity
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=169G
#SBATCH --partition=bigmem
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/attachment_ambiguity/facebook-llama-30B_attachment_ambiguity.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'facebook/llama/30B' \
	--test_file data/attachment_ambiguity/attachment_ambiguity.txt.gz \
	--tokenizer_name facebook/llama/tokenizer.model