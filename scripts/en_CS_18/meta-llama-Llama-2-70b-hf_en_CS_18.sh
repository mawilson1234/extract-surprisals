#!/bin/bash

#SBATCH --job-name=meta-llama-Llama-2-70b-hf-en_CS_18
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=288G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts\en_CS_18\meta-llama-Llama-2-70b-hf_en_CS_18.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'meta-llama/Llama-2-70b-hf' \
	--test_file data/en_CS_18/en_CS_18.txt.gz \
	--save_tmp \
	--use_auth_token ~/.hf_auth_token