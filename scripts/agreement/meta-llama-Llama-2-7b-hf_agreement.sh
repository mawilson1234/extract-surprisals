#!/bin/bash

#SBATCH --job-name=meta-llama-Llama-2-7b-hf-agreement
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/agreement/meta-llama-Llama-2-7b-hf_agreement.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'meta-llama/Llama-2-7b-hf' \
	--test_file data/agreement/agreement.txt.gz \
	--token ~/.hf_auth_token