#!/bin/bash

#SBATCH --job-name=distilbert-base-cased-agreement
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/agreement/distilbert-base-cased_agreement.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path distilbert-base-cased \
	--test_file data/agreement/agreement.txt.gz