#!/bin/bash

#SBATCH --job-name=google-bert_uncased_L-10_H-768_A-12-fillers
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts\fillers\google-bert_uncased_L-10_H-768_A-12_fillers.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'google/bert_uncased_L-10_H-768_A-12' \
	--test_file data/fillers/fillers.txt.gz