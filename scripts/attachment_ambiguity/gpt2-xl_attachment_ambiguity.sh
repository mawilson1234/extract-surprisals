#!/bin/bash

#SBATCH --job-name=gpt2-xl-attachment_ambiguity
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=64G
#SBATCH --partition=day
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/attachment_ambiguity/gpt2-xl_attachment_ambiguity.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path gpt2-xl \
	--test_file data/attachment_ambiguity/attachment_ambiguity.txt.gz