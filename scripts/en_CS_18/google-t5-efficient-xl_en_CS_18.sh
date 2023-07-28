#!/bin/bash

#SBATCH --job-name=google-t5-efficient-xl-en_CS_18
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=02:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_CS_18/google-t5-efficient-xl_en_CS_18.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'google/t5-efficient-xl' \
	--test_file data/en_CS_18/en_CS_18.txt.gz