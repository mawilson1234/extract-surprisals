#!/bin/bash

#SBATCH --job-name=bert-base-uncased-agreement
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=02:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/agreement/bert-base-uncased_agreement.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path bert-base-uncased \
	--test_file data/agreement/agreement.txt.gz