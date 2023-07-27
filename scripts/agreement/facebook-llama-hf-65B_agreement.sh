#!/bin/bash

#SBATCH --job-name=facebook-llama-hf-65B-agreement
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=288G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/agreement/facebook-llama-hf-65B_agreement.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'facebook/llama-hf/65B' \
	--test_file data/agreement/agreement.txt.gz \
	--save_tmp