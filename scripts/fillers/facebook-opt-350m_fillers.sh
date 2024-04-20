#!/bin/bash

#SBATCH --job-name=facebook-opt-350m-fillers
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts\fillers\facebook-opt-350m_fillers.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'facebook/opt-350m' \
	--test_file data/fillers/fillers.txt.gz