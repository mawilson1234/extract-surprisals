#!/bin/bash

#SBATCH --job-name=google-electra-large-generator-agreement
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/agreement/google-electra-large-generator_agreement.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'google/electra-large-generator' \
	--test_file data/agreement/agreement.txt.gz