#!/bin/bash

#SBATCH --job-name=gpt2-large-classic_gp
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=02:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/classic_gp/gpt2-large_classic_gp.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path gpt2-large \
	--test_file data/classic_gp/classic_gp.txt.gz