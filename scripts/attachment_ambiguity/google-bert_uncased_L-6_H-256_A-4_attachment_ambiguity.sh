#!/bin/bash

#SBATCH --job-name=google-bert_uncased_L-6_H-256_A-4-attachment_ambiguity
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts\attachment_ambiguity\google-bert_uncased_L-6_H-256_A-4_attachment_ambiguity.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'google/bert_uncased_L-6_H-256_A-4' \
	--test_file data/attachment_ambiguity/attachment_ambiguity.txt.gz