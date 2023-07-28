#!/bin/bash

#SBATCH --job-name=yanaiela-roberta-base-epoch_8-attachment_ambiguity
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=02:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/attachment_ambiguity/yanaiela-roberta-base-epoch_8_attachment_ambiguity.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'yanaiela/roberta-base-epoch_8' \
	--test_file data/attachment_ambiguity/attachment_ambiguity.txt.gz