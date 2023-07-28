#!/bin/bash

#SBATCH --job-name=mueller-c4-100m_withchildes-en_WD_23
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=02:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/en_WD_23/mueller-c4-100m_withchildes_en_WD_23.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'mueller/c4-100m_withchildes' \
	--test_file data/en_WD_23/en_WD_23.txt.gz