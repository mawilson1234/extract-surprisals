#!/bin/bash

#SBATCH --job-name=google-multiberts-seed_1-step_700k-en_LM_22
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_LM_22/google-multiberts-seed_1-step_700k_en_LM_22.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'google/multiberts-seed_1-step_700k' \
	--test_file data/en_LM_22/en_LM_22.txt.gz