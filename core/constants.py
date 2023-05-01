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

CASES: Set[str] = {'uncased', 'cased'}

MASKED_LANGUAGE_MODELS: Set[str] = (
	{f'distilbert-base-{case}' for case in CASES} |
	{f'bert-{size}-{case}' for case in CASES for size in {'base', 'large'}} |
	{f'roberta-{size}' for size in {'base', 'large'}} |
	{f'phueb/BabyBERTa-{i}' for i in range(1,4)} |
	{'distilroberta-base'} |
	{
		f'albert-{size}-{version}' 
		for size in 
		{'base', 'large', 'xlarge', 'xxlarge'} 
			for version in 
			{'v1', 'v2'}
	} |
	# these models do not currently work (04/2023)
	# see https://github.com/huggingface/transformers/pull/18674
	# {
	#	f'microsoft/deberta-v3-{size}'
	#	for size in 
	#	{'xsmall', 'small', 'base', 'large'}
	# } |
	# {
	#	f'microsoft/deberta-{size}'
	#	for size in 
	#	{'base', 'large', 'xlarge'}
	# } |
	# {
	#	f'microsoft/deberta-v2-{size}' 
	#	for size in 
	#	{'xlarge', 'xxlarge'}
	# } |
	{
		f'google/electra-{size}-generator'
		for size in 
		{'small', 'base', 'large'}
	} |
	{
		f'google/multiberts-seed_{i}'
		for i in range(25)
	} |
	{
		f'google/multiberts-seed_{i}-step_{n}k'
		for i in range(5)
		for n in set(range(0,200,20)) | set(range(200,2001,100))
	} |
	{
		f'google/bert_uncased_L-{l}_H-{h}_A-{a}' 
		for l in {2, 4, 6, 8, 10, 12} 
		for h, a in zip((128, 256, 512, 768), (2, 4, 8, 12))
	} |
	{
		f'yanaiela/roberta-base-epoch_{n}'
		for n in range(84)
	}
)

NEXT_WORD_MODELS: Set[str] = (
	OPT_MODELS |
	GPT2_MODELS |
	LLAMA_MODELS
)

MUELLER_T5_MODELS: Set[str] = ({
	f'mueller/{m}' for m in
		{f'{pfx}-1m' for pfx in 
			{'babyt5', 'c4', 'wikit5', 'simplewiki'}} |
		{'babyt5-5m'} |
		{m for pfx in 
			{'c4', 'wikit5', 'simplewiki'}
				for m in 
				{f'{pfx}-{i}m' for i in {10, 100}}
		} |
		{m for pfx in 
			{'c4', 'wikit5'}
				for m in
				{f'{pfx}-{i}' for i in {'100m_withchildes', '1b'}}
		}
})

GOOGLE_T5_MODELS: Set[str] = (
	{f'google/t5-efficient-{size}' 
		for size in 
		{'tiny', 'mini', 'small', 'base', 'large', 'xl', 'xxl'}
	} | 
	{f'google/t5-efficient-base-{ablation}'
		for ablation in 
		{f'dl{i}' for i in range(2,9,2)} |
		{f'el{i}' for i in range(2,9,2)} |
		{f'nl{i}' for i in (2**i for i in range(1,4,1))} |
		{f'nh{i}' for i in range(8, 33, 8)}
	} |
	{f'google/t5-efficient-mini-{ablation}'
		for ablation in 
		{f'nl{i}' for i in {6, 8, 12, 24}}
	}
)

T5_MODELS: Set[str] = (
	MUELLER_T5_MODELS |
	GOOGLE_T5_MODELS
)

SEQ2SEQ_MODELS: Set[str] = (
	T5_MODELS
)

ALL_MODELS: Set[str] = (
	NEXT_WORD_MODELS |
	MASKED_LANGUAGE_MODELS |
	SEQ2SEQ_MODELS
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
	if 'BabyBERTa' in model_name_or_path:
		tokenizer_kwargs = {**tokenizer_kwargs, 'add_prefix_space': True}
	
	# these ones can't be used fast,
	# since it causes problems
	if any(s in model_name_or_path for s in {'deberta-v3', 'opt-'}):
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
		'If you would like to add it, you should add it to `SEQ2SEQ_MODELS` '
		'for T5 models (evaluated using a conditional generation task), '
		'`MASKED_LANGUAGE_MODELS` for models that should be evaluated using '
		'a masked language modeling task, or `NEXT_WORD_MODELS` for models that '
		'should be evaluated using a language modeling task. (Encoding-decoder models '
		'beside T5 models are not currently supported.)'
	)