#!/bin/bash

#SBATCH --job-name=meta-llama-Llama-2-70b-hf-en_LCLB_23
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=288G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_LCLB_23/meta-llama-Llama-2-70b-hf_en_LCLB_23.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'meta-llama/Llama-2-70b-hf' \
	--test_file data/en_LCLB_23/en_LCLB_23.txt.gz \
	--save_tmp \
	--use_auth_token ~/.hf_auth_token