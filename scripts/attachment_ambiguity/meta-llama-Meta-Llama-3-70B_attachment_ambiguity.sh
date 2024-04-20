#!/bin/bash

#SBATCH --job-name=meta-llama-Meta-Llama-3-70B-attachment_ambiguity
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=288G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/attachment_ambiguity/meta-llama-Meta-Llama-3-70B_attachment_ambiguity.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'meta-llama/Meta-Llama-3-70B' \
	--test_file data/attachment_ambiguity/attachment_ambiguity.txt.gz \
	--save_tmp \
	--token ~/.hf_auth_token