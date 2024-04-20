#!/bin/bash

#SBATCH --job-name=google-multiberts-seed_4-step_1000k-classic_gp
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts\classic_gp\google-multiberts-seed_4-step_1000k_classic_gp.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'google/multiberts-seed_4-step_1000k' \
	--test_file data/classic_gp/classic_gp.txt.gz