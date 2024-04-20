#!/bin/bash

#SBATCH --job-name=mueller-c4-10m-fillers
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/fillers/mueller-c4-10m_fillers.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'mueller/c4-10m' \
	--test_file data/fillers/fillers.txt.gz