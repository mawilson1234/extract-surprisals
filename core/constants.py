from typing import Set, Dict, List

GPT2_MODELS: Set[str] = (
	{'gpt2'} |
	{f'gpt2-{s}' 
		for s in 
		{'medium', 'large', 'xl'}
	}
)

OPT_MODELS: Set[str] = (
	{f'facebook/opt-{i}m' for i in 
		{125, 350}
	} |
	{f'facebook/opt-{i}b' for i in 
		{1.3, 2.7, 6.7, 13, 30, 66, 175}
	}
)

LLAMA_MODELS: Set[str] = (
	{f'facebook/llama/{i}B' for i in
		{7, 13, 30, 65}
	}
)

ALL_MODELS: Set[str] = (
	OPT_MODELS |
	GPT2_MODELS |
	LLAMA_MODELS
)

import os
DATASETS: Set[str] = set(
	os.listdir(os.path.join(os.path.dirname(__file__), '..', 'data'))
)

def get_tokenizer_kwargs(model_name_or_path: str) -> Dict:
	# gets the appropriate kwargs for the tokenizer
	# this provides us a single place where we can deal
	# with idiosyncrasies of specific tokenizers
	tokenizer_kwargs = {'pretrained_model_name_or_path': model_name_or_path}
	
	# these ones can't be used fast,
	# since it causes problems
	if any(s in model_name_or_path for s in {'opt-'}):
		tokenizer_kwargs = {**tokenizer_kwargs, 'use_fast': False}
	
	# opt-175b doesn't come with a HF tokenizer,
	# but it uses the same one as the smaller models
	# which are available on HF
	if 'opt-175b' in model_name_or_path:
		tokenizer_kwargs = {**tokenizer_kwargs, 'pretrained_model_name_or_path': 'facebook/opt-125m'}
	
	if 'llama' in model_name_or_path:
		tokenizer_kwargs = {**tokenizer_kwargs, 'pretrained_model_name_or_path': 'facebook/llama/tokenizer.model'}
	
	return tokenizer_kwargs

def model_not_supported_message(model_name_or_path: str) -> str:
	return (
		f'{model_name_or_path!r} has not been classified in `constants.py`. '
		'If you would like to add it, you should add it to `ALL_MODELS`, after '
		'making sure to deal with any idiosyncratic tokenizer options in '
		'`get_tokenizer_kwargs`.'
	)