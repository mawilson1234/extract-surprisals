#!/bin/bash

#SBATCH --job-name=meta-llama-Meta-Llama-3-70B-fillers
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=288G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/fillers/meta-llama-Meta-Llama-3-70B_fillers.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'meta-llama/Meta-Llama-3-70B' \
	--test_file data/fillers/fillers.txt.gz \
	--save_tmp \
	--use_auth_token ~/.hf_auth_token