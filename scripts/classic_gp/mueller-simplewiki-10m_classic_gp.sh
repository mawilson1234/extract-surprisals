#!/bin/bash

#SBATCH --job-name=mueller-simplewiki-10m-classic_gp
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=48G
#SBATCH --partition=day
#SBATCH --time=02:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate ext-surp

echo "Running script scripts/classic_gp/mueller-simplewiki-10m_classic_gp.sh"
echo ""

python core/extract_surprisals.py \
	--model_name_or_path 'mueller/simplewiki-10m' \
	--test_file data/classic_gp/classic_gp.txt.gz