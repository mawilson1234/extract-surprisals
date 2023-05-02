#!/bin/bash

#SBATCH --job-name=albert-xlarge-v1-classic_gp
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/classic_gp/albert-xlarge-v1_classic_gp.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path albert-xlarge-v1 \
	--test_file data/classic_gp/classic_gp.txt.gz