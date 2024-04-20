#!/bin/bash

#SBATCH --job-name=facebook-opt-30b-attachment_ambiguity
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=169G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts\attachment_ambiguity\facebook-opt-30b_attachment_ambiguity.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'facebook/opt-30b' \
	--test_file data/attachment_ambiguity/attachment_ambiguity.txt.gz \
	--save_tmp