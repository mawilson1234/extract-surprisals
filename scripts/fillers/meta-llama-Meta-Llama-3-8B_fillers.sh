#!/bin/bash

#SBATCH --job-name=meta-llama-Meta-Llama-3-8B-fillers
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/fillers/meta-llama-Meta-Llama-3-8B_fillers.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'meta-llama/Meta-Llama-3-8B' \
	--test_file data/fillers/fillers.txt.gz \
	--token ~/.hf_auth_token