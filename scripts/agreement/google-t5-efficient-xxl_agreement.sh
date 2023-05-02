#!/bin/bash

#SBATCH --job-name=google-t5-efficient-xxl-agreement
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=169G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/agreement/google-t5-efficient-xxl_agreement.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'google/t5-efficient-xxl' \
	--test_file data/agreement/agreement.txt.gz \
	--save_tmp