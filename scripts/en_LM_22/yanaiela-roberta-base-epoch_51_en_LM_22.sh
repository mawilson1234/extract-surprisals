#!/bin/bash

#SBATCH --job-name=yanaiela-roberta-base-epoch_51-en_LM_22
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_LM_22/yanaiela-roberta-base-epoch_51_en_LM_22.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'yanaiela/roberta-base-epoch_51' \
	--test_file data/en_LM_22/en_LM_22.txt.gz