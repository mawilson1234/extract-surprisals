#!/bin/bash

#SBATCH --job-name=meta-llama-Llama-2-13b-hf-fillers
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=78G
#SBATCH --partition=day
#SBATCH --time=16:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/fillers/meta-llama-Llama-2-13b-hf_fillers.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'meta-llama/Llama-2-13b-hf' \
	--test_file data/fillers/fillers.txt.gz \
	--token ~/.hf_auth_token