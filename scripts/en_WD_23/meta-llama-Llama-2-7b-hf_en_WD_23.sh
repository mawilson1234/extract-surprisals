#!/bin/bash

#SBATCH --job-name=meta-llama-Llama-2-7b-hf-en_WD_23
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_WD_23/meta-llama-Llama-2-7b-hf_en_WD_23.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'meta-llama/Llama-2-7b-hf' \
	--test_file data/en_WD_23/en_WD_23.txt.gz \
	--token ~/.hf_auth_token