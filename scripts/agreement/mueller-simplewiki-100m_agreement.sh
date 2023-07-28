#!/bin/bash

#SBATCH --job-name=mueller-simplewiki-100m-agreement
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=02:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/agreement/mueller-simplewiki-100m_agreement.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'mueller/simplewiki-100m' \
	--test_file data/agreement/agreement.txt.gz