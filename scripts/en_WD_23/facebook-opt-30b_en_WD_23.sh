#!/bin/bash

#SBATCH --job-name=facebook-opt-30b-en_WD_23
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=169G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_WD_23/facebook-opt-30b_en_WD_23.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'facebook/opt-30b' \
	--test_file data/en_WD_23/en_WD_23.txt.gz \
	--save_tmp