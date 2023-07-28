#!/bin/bash

#SBATCH --job-name=phueb-BabyBERTa-1-agreement
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=02:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/agreement/phueb-BabyBERTa-1_agreement.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'phueb/BabyBERTa-1' \
	--test_file data/agreement/agreement.txt.gz