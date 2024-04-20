#!/bin/bash

#SBATCH --job-name=meta-llama-Meta-Llama-3-70B-en_WD_23
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=288G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_WD_23/meta-llama-Meta-Llama-3-70B_en_WD_23.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'meta-llama/Meta-Llama-3-70B' \
	--test_file data/en_WD_23/en_WD_23.txt.gz \
	--save_tmp \
	--use_auth_token ~/.hf_auth_token