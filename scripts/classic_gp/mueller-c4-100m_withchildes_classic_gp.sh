#!/bin/bash

#SBATCH --job-name=mueller-c4-100m_withchildes-classic_gp
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/classic_gp/mueller-c4-100m_withchildes_classic_gp.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'mueller/c4-100m_withchildes' \
	--test_file data/classic_gp/classic_gp.txt.gz