#!/bin/bash

#SBATCH --job-name=facebook-opt-30b-classic_gp
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=169G
#SBATCH --partition=bigmem
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/classic_gp/facebook-opt-30b_classic_gp.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'facebook/opt-30b' \
	--test_file data/classic_gp/classic_gp.txt.gz