#!/bin/bash

#SBATCH --job-name=meta-llama-Llama-2-70b-hf-classic_gp
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=288G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/classic_gp/meta-llama-Llama-2-70b-hf_classic_gp.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'meta-llama/Llama-2-70b-hf' \
	--test_file data/classic_gp/classic_gp.txt.gz \
	--tokenizer_name facebook/llama/tokenizer.model \
	--save_tmp