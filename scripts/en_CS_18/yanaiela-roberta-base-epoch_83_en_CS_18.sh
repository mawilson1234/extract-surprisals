#!/bin/bash

#SBATCH --job-name=yanaiela-roberta-base-epoch_83-en_CS_18
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_CS_18/yanaiela-roberta-base-epoch_83_en_CS_18.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'yanaiela/roberta-base-epoch_83' \
	--test_file data/en_CS_18/en_CS_18.txt.gz