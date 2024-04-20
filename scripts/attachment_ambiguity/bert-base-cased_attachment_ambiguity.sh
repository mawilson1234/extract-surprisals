#!/bin/bash

#SBATCH --job-name=bert-base-cased-attachment_ambiguity
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts\attachment_ambiguity\bert-base-cased_attachment_ambiguity.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path bert-base-cased \
	--test_file data/attachment_ambiguity/attachment_ambiguity.txt.gz