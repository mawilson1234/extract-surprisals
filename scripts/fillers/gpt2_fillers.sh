#!/bin/bash

#SBATCH --job-name=gpt2-fillers
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=02:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/fillers/gpt2_fillers.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path gpt2 \
	--test_file data/fillers/fillers.txt.gz