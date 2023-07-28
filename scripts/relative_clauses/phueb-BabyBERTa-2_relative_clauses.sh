#!/bin/bash

#SBATCH --job-name=phueb-BabyBERTa-2-relative_clauses
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=02:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/relative_clauses/phueb-BabyBERTa-2_relative_clauses.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'phueb/BabyBERTa-2' \
	--test_file data/relative_clauses/relative_clauses.txt.gz