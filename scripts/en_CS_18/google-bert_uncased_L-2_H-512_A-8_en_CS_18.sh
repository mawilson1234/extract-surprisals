#!/bin/bash

#SBATCH --job-name=google-bert_uncased_L-2_H-512_A-8-en_CS_18
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_CS_18/google-bert_uncased_L-2_H-512_A-8_en_CS_18.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'google/bert_uncased_L-2_H-512_A-8' \
	--test_file data/en_CS_18/en_CS_18.txt.gz