#!/bin/bash

#SBATCH --job-name=gpt2-relative_clauses
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=64G
#SBATCH --partition=day
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/relative_clauses/gpt2_relative_clauses.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path gpt2 \
	--test_file data/relative_clauses/relative_clauses.txt.gz