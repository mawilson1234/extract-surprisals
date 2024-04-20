#!/bin/bash

#SBATCH --job-name=meta-llama-Meta-Llama-3-70B-agreement
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=288G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/agreement/meta-llama-Meta-Llama-3-70B_agreement.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'meta-llama/Meta-Llama-3-70B' \
	--test_file data/agreement/agreement.txt.gz \
	--save_tmp \
	--token ~/.hf_auth_token