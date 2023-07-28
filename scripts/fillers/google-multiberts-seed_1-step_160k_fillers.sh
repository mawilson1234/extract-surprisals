#!/bin/bash

#SBATCH --job-name=google-multiberts-seed_1-step_160k-fillers
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=02:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/fillers/google-multiberts-seed_1-step_160k_fillers.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'google/multiberts-seed_1-step_160k' \
	--test_file data/fillers/fillers.txt.gz