#!/bin/bash

#SBATCH --job-name=meta-llama-Llama-2-70b-hf-fillers
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=320G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/fillers/meta-llama-Llama-2-70b-hf_fillers.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'meta-llama/Llama-2-70b-hf' \
	--test_file data/fillers/fillers.txt.gz \
	--tokenizer_name facebook/llama/tokenizer.model \
	--save_tmp