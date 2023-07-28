#!/bin/bash

#SBATCH --job-name=google-electra-base-generator-fillers
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=02:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/fillers/google-electra-base-generator_fillers.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'google/electra-base-generator' \
	--test_file data/fillers/fillers.txt.gz