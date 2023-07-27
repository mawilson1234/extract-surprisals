import os
import re

from tqdm import tqdm
from typing import Set, Dict
from core.constants import *

SCRIPT_TEMPLATE: str = '\n'.join([
	'#!/bin/bash',
	'',
	'#SBATCH --job-name={MODEL_BASENAME}-{DATASET}',
	'#SBATCH --output=joblogs/%x_%j.txt',
	'#SBATCH --mem={MEM}',
	'#SBATCH --partition={PART}',
	'#SBATCH --time={TIME}',
	'#SBATCH --mail-type=END,FAIL,INVALID_DEPEND',
	'',
	'module load miniconda',
	'',
	'source activate ext-surp',
	'',
	'echo "Running script {FILENAME}"',
	'echo ""',
	'',
	'python core/extract_surprisals.py \\',
	'\t--model_name_or_path {MODEL_FULLNAME} \\',
	'\t--test_file data/{DATASET}/{DATASET}.txt.gz',
])

# these models need to be run with special options
BIG_MODELS: Dict[str, Dict[str,str]] = {
	'google/t5-efficient-xxl':	{'mem': '169G', 'time': '01-00:00:00', 'partition': 'bigmem'},
	'facebook/opt-2.7b':		{'mem': '48G',  'time': '08:00:00',    'partition': 'day'},
	'facebook/opt-6.7b':		{'mem': '48G',  'time': '08:00:00',    'partition': 'day'},
	'facebook/opt-13b':			{'mem': '78G',  'time': '16:00:00',    'partition': 'day'},
	'facebook/opt-30b':			{'mem': '169G', 'time': '01-00:00:00', 'partition': 'bigmem'},
	'facebook/opt-66b':			{'mem': '288G', 'time': '01-00:00:00', 'partition': 'bigmem'},
	'facebook/opt-175b':		{'mem': '700G', 'time': '01-00:00:00', 'partition': 'bigmem'},
	'facebook/llama-hf/7B':		{'mem': '48G',  'time': '08:00:00',    'partition': 'day'},
	'facebook/llama-hf/13B':	{'mem': '78G',  'time': '16:00:00',    'partition': 'day'},
	'facebook/llama-hf/30B':	{'mem': '169G', 'time': '01-00:00:00', 'partition': 'bigmem'},
	'facebook/llama-hf/65B':	{'mem': '288G', 'time': '01-00:00:00', 'partition': 'bigmem'},
	'meta-llama/Llama-2-7b-hf':	{'mem': '48G',  'time': '08:00:00',    'partition': 'day'},
	'meta-llama/Llama-2-13b-hf':{'mem': '78G',  'time': '16:00:00',    'partition': 'day'},
	'meta-llama/Llama-2-70b-hf':{'mem': '288G', 'time': '01-00:00:00', 'partition': 'bigmem'},
}

# these models need more than a day to run, 
# so we use this to add an option to save tmp files
# from which we can resume evaluation
NEED_MORE_THAN_ONE_DAY: Set[str] = (
	{k for k in BIG_MODELS.keys() if BIG_MODELS[k]['time'].startswith('01-')}	
)

def create_scripts() -> None:
	with tqdm(total=len(DATASETS) * len(ALL_MODELS)) as pbar:
		for dataset in DATASETS:
			script_dirname = dataset
			os.makedirs(os.path.join('scripts', script_dirname), exist_ok=True)
			for model in ALL_MODELS:
				
				script = SCRIPT_TEMPLATE
				
				if model in LLAMA_MODELS and not '-hf' in model:
					script += ' \\\n\t--tokenizer_name facebook/llama/tokenizer.model'
				
				if model in NEED_MORE_THAN_ONE_DAY:
					script += ' \\\n\t--save_tmp'
				
				if 'Llama-2' in model:
					script += ' \\\n\t--use_auth_token ~/.hf_auth_token'
				
				# deal with slashes in model names
				model_basename = re.sub(r'[\\/]', '-', model)
				or_model = model
				if model_basename != model:
					model = f"'{model}'"
				
				script_filename = os.path.join('scripts', script_dirname, f'{model_basename}_{dataset}.sh')
				
				script = script.format(
					MODEL_BASENAME=model_basename,
					DATASET=dataset,
					MODEL_FULLNAME=model,
					MEM=BIG_MODELS.get(or_model, {}).get('mem', '48G'),
					PART=BIG_MODELS.get(or_model, {}).get('partition', 'day'),
					TIME=BIG_MODELS.get(or_model, {}).get('time', '01:00:00'),
					FILENAME=script_filename,
				)
				
				with open(script_filename, 'wt', encoding='utf8') as out_file:
					_ = out_file.write(script)
				
				pbar.update(1)

if __name__ == '__main__':
	create_scripts()
