#!/bin/bash

#SBATCH --job-name=facebook-opt-175b-classic_gp
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=700G
#SBATCH --partition=bigmem
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts\classic_gp\facebook-opt-175b_classic_gp.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'facebook/opt-175b' \
	--test_file data/classic_gp/classic_gp.txt.gz \
	--save_tmp