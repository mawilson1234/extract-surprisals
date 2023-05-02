#!/bin/bash

#SBATCH --job-name=google-bert_uncased_L-2_H-128_A-2-relative_clauses
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/relative_clauses/google-bert_uncased_L-2_H-128_A-2_relative_clauses.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'google/bert_uncased_L-2_H-128_A-2' \
	--test_file data/relative_clauses/relative_clauses.txt.gz