#!/bin/bash

#SBATCH --job-name=google-t5-efficient-base-nh16-en_WD_23
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_WD_23/google-t5-efficient-base-nh16_en_WD_23.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'google/t5-efficient-base-nh16' \
	--test_file data/en_WD_23/en_WD_23.txt.gz