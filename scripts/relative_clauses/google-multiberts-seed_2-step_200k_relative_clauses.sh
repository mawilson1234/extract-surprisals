#!/bin/bash

#SBATCH --job-name=google-multiberts-seed_2-step_200k-relative_clauses
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/relative_clauses/google-multiberts-seed_2-step_200k_relative_clauses.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'google/multiberts-seed_2-step_200k' \
	--test_file data/relative_clauses/relative_clauses.txt.gz