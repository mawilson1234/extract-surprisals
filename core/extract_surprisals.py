########################################################################################################
# This script is (heavily) adapted from a script by HuggingFace Inc. 								   #
# It has been modified for use with a language                                                         #
# modeling tasks by Michael Wilson (2022-2023).			   											   #
########################################################################################################
#
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

# Setup logging
logging.basicConfig(
	format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
	datefmt="%m/%d/%Y %H:%M:%S",
	level=logging.INFO
)
logger = logging.getLogger(__name__)

import os
import re
import sys
import gzip
import json
import time
import torch
import datasets
datasets.logging.set_verbosity_error()

import transformers
transformers.utils.logging.set_verbosity_error()

import pandas as pd
import torch.nn.functional as F

from glob import glob
from tqdm import tqdm
from typing import *
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from dataclasses import dataclass, field
from transformers import (
	AutoConfig,
	AutoTokenizer,
	HfArgumentParser,
	AutoModelForCausalLM,
	AutoModelForMaskedLM,
	T5ForConditionalGeneration
)

if __name__ == '__main__':
	from constants import *
	from llama import (
		ModelArgs,
		Transformer,
		Tokenizer,
		LLaMA
	)
else:
	from .constants import *
	from .llama import (
		ModelArgs, 
		Transformer, 
		Tokenizer, 
		LLaMA
	)

# this is a dummy to let us 
# put attributes in the expected 
# location for LLaMA models
class Config():
	pass

@dataclass
class ModelArguments:
	'''Arguments pertaining to which model/config/tokenizer we are going to evaluate.'''
	model_name_or_path: str = field(
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	
	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
	)
	
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
	)
	
	tokenizer_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
	)
	
	use_fast_tokenizer: Optional[bool] = field(
		default=True,
		metadata={"help": "Whether to use one of the fast tokenizers (backed by the tokenizers library) or not."},
	)
	
	model_revision: str = field(
		default="main",
		metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
	)
	
	use_auth_token: str = field(
		default=False,
		metadata={
			"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
			"with private models)."
		},
	)
	
	def __post_init__(self):
		self.config_name = self.config_name or self.model_name_or_path
		self.tokenizer_name = self.tokenizer_name or self.model_name_or_path
		self.tokenizer_kwargs = get_tokenizer_kwargs(self.tokenizer_name)
		
		self.tokenizer_name = self.tokenizer_kwargs['pretrained_model_name_or_path']
		del self.tokenizer_kwargs['pretrained_model_name_or_path']
		
		if 'use_fast' in self.tokenizer_kwargs:
			self.use_fast_tokenizer = self.tokenizer_kwargs['use_fast']
			del self.tokenizer_kwargs['use_fast']
		
		self.use_auth_token = None if not self.use_auth_token else self.use_auth_token
		if self.use_auth_token is not None and os.path.isfile(os.path.expanduser(self.use_auth_token)):
			with open(os.path.expanduser(self.use_auth_token), 'rt') as in_file:
				self.use_auth_token = in_file.read().strip()
			
		self.from_flax = self.model_name_or_path in MUELLER_T5_MODELS

@dataclass
class DataArguments:
	'''Arguments pertaining to what data we are going to input our model for evaluation.'''
	output_dir: str = field(
		default=None,
		metadata={"help": "Where to store the results of the evaluation."}
	)
	
	test_file: str = field(
		default=None,
		metadata={"help": "The test data file to evaluate model predictions for."},
	)
	
	overwrite_cache: bool = field(
		default=False, 
		metadata={"help": "Overwrite the cached evaluation set"}
	)
	
	preprocessing_num_workers: Optional[int] = field(
		default=None,
		metadata={"help": "The number of processes to use for the preprocessing."},
	)
	
	max_source_length: Optional[int] = field(
		default=1024,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	
	per_device_test_batch_size: int = field(
		default=16,
		metadata={
			"help": "The batch size for evaluation data."
		}
	)
		
	max_test_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes, truncate the number of test examples to this "
			"value if set."
		},
	)
	
	save_tmp: Optional[bool] = field(
		default=False,
		metadata={
			"help": "Whether to store temporary files for each batch of results. These can "
			"be used to resume evaluation if it is interrupted. Useful for large models "
			"which may not be able to finish on time within cluster resource limits."
		},
	)
	
	def __post_init__(self):
		if self.test_file is None:
			raise ValueError('Need a test file.')
		
		extension = '.'.join(self.test_file.split('.')[-2:])
		assert extension == 'txt.gz', "`test_file` should be a a txt.gz file."
		
		if self.output_dir is None:
			self.output_dir = os.path.join('outputs', os.path.basename(self.test_file).replace('.txt.gz', ''))

def parse_cl_arguments(*args: Tuple) -> Tuple:
	'''
	Parse command line arguments into ModelArguments and DataArguments.
	See ModelArguments and DataArguments for details.
	'''
	parser = HfArgumentParser(args)
	model_args, data_args = parser.parse_args_into_dataclasses()
	
	return model_args, data_args

def load_model(model_name_or_path: str, *args, **kwargs):
	'''
	Loads the model using the appropriate function.
	'''
	if model_name_or_path in NEXT_WORD_MODELS:
		return AutoModelForCausalLM.from_pretrained(
			model_name_or_path, *args, **kwargs
		)
	
	if model_name_or_path in MASKED_LANGUAGE_MODELS:
		return AutoModelForMaskedLM.from_pretrained(
			model_name_or_path, *args, **kwargs
		)
	
	if model_name_or_path in T5_MODELS:
		return T5ForConditionalGeneration.from_pretrained(
			model_name_or_path, *args, **kwargs
		)
	
	raise ValueError(model_not_supported_message(model_name_or_path))

def load_HF_tokenizer_and_model(model_args: ModelArguments) -> Tuple:
	'''Loads the tokenizer and model as specified in model_args.'''
	if 'llama' in model_args.model_name_or_path and not '-hf' in model_args.model_name_or_path:
		raise ValueError(
			'`load_tokenizer_and_model` should only be used '
			'for models on the Hugging Face hub. For LLaMA '
			'use `load_llama_tokenizer` and `load_llama` instead.'
		)
	
	config = AutoConfig.from_pretrained(
		model_args.config_name,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
		use_auth_token=model_args.use_auth_token
	)
	
	tokenizer = AutoTokenizer.from_pretrained(
		model_args.tokenizer_name,
		cache_dir=model_args.cache_dir,
		use_fast=model_args.use_fast_tokenizer,
		revision=model_args.model_revision,
		use_auth_token=model_args.use_auth_token,
		**model_args.tokenizer_kwargs,
	)
	
	if tokenizer.mask_token_id is None:
		if model_args.model_name_or_path in T5_MODELS:
			tokenizer.mask_token_id = tokenizer.get_vocab()['<extra_id_0>']
	
	if tokenizer.name_or_path in LLAMA_MODELS and '-hf' in tokenizer.name_or_path:
		tokenizer.add_special_tokens({'pad_token': '[PAD]'})
	
	model = load_model(
		model_args.model_name_or_path,
		from_flax=model_args.from_flax,
		config=config,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
		use_auth_token=model_args.use_auth_token,
	)
	
	if model.name_or_path in LLAMA_MODELS and '-hf' in model.name_or_path:
		model.resize_token_embeddings(len(tokenizer))
	
	if model_args.model_name_or_path in GPT2_MODELS:
		tokenizer.pad_token = tokenizer.eos_token
		model.config.pad_token_id = model.config.eos_token_id
		tokenizer.bos_token = tokenizer.eos_token
		model.config.bos_token_id = model.config.eos_token_id
	
	return tokenizer, model

def load_llama_tokenizer(
	tokenizer_path: str,
) -> Tokenizer:
	'''
	Loads the LLaMA tokenizer.
	We do this separately so we can tokenize
	the dataset before loading the model, and thus
	automatically adjust the maximum sequence length.
	'''
	tokenizer = Tokenizer(model_path=tokenizer_path)
	setattr(tokenizer, 'name_or_path', tokenizer_path)
	setattr(tokenizer, 'pad_token_id', tokenizer.pad_id)
	
	return tokenizer

def load_llama(
	ckpt_dir: str,
	tokenizer: Tokenizer,
	max_seq_len: int,
	max_batch_size: int = 1,
) -> LLaMA:
	'''
	Loads and returns a LLaMA generator.
	
	params:
		ckpt_dir (str): the location of the directory containing the LLaMA checkpoint
		tokenizer (Tokenizer): the tokenizer for the model
		max_seq_len (int): the maximum sequence length that can be generated.
						   must be at least the length of the longest (tokenized) input sequence
		max_batch_size (int): at most this many examples will be run in a batch
	
	returns:
		LLaMA: the LLaMA generator
	'''
	checkpoints = sorted(Path(ckpt_dir).glob('*.pth'))
	
	with open(Path(ckpt_dir)/'params.json', 'r') as f:
		params = json.loads(f.read())
	
	model_args = ModelArgs(
		max_seq_len=max_seq_len, 
		max_batch_size=max_batch_size,
		**params
	)
	
	model_args.vocab_size = tokenizer.n_words
	
	model = Transformer(model_args)
	
	# Original copyright by tloen
	# https://github.com/tloen/llama-int8/blob/main/example.py
	key_to_dim = {
		"w1": 0,
		"w2": -1,
		"w3": 0,
		"wo": -1,
		"wq": 0,
		"wk": 0,
		"wv": 0,
		"output": 0,
		"tok_embeddings": -1,
		"ffn_norm": None,
		"attention_norm": None,
		"norm": None,
		"rope": None,
	}
	
	logger.info('Loading checkpoints to model...')
	for i, ckpt in tqdm(enumerate(checkpoints), total=len(checkpoints)):
		checkpoint = torch.load(ckpt, map_location='cpu')
		for parameter_name, parameter in model.named_parameters():
			short_name = parameter_name.split(".")[-2]
			if key_to_dim[short_name] is None and i == 0:
				parameter.data = checkpoint[parameter_name]
			elif key_to_dim[short_name] == 0:
				size = checkpoint[parameter_name].size(0)
				parameter.data[size * i: size * (i + 1), :] = checkpoint[
					parameter_name
				]
			elif key_to_dim[short_name] == -1:
				size = checkpoint[parameter_name].size(-1)
				parameter.data[:, size * i: size * (i + 1)] = checkpoint[
					parameter_name
				]
			del checkpoint[parameter_name]
		del checkpoint
	
	model.to('cpu')
	
	generator = LLaMA(model, tokenizer)
	
	setattr(generator, 'config', Config())
	setattr(generator.config, 'name_or_path', ckpt_dir)
	setattr(generator, 'name_or_path', ckpt_dir)
	setattr(generator, 'parameters', generator.model.parameters())
	
	return generator

def expand_with_masks(
	inputs: torch.Tensor, 
	dataset: Dataset, 
	mask_id: int, 
	dont_mask: List[int], 
	metadata: List[Dict] = None
) -> torch.Tensor:
	'''
	Expands a list of inputs by generating a new set of inputs
	where each example replaces exactly one token with the mask
	token, ignoring ignore_tokens (= special tokens).
	'''
	if metadata is None:
		metadata = [{}] * inputs['input_ids'].shape[0]
	
	def _expand_with_masks(example: torch.Tensor, text_example: str, metadata: Dict, mask_id: int, dont_mask: List[int]) -> torch.Tensor:
		'''
		Expands a single example to a list of examples, constructed
		by replacing each non-special token (=ignore_tokens) with the mask token id.
		This allows us to get surprisal for each position in the sentence from
		a masked language model without telling it what the token in that
		position is.
		'''
		ignore_mask = [x not in dont_mask for x in example['input_ids']]
		expanded_length = sum(ignore_mask)
		
		expanded = dict()
		for k in example:
			expanded[k] = example[k].expand(1, example[k].shape[0]).repeat(expanded_length, 1)
		
		expanded_metadata = [metadata] * expanded_length
		expanded_text = [text_example] * expanded_length
		
		starting_id = ignore_mask.index(True)
		
		token_index = starting_id
		for row in expanded['input_ids']:
			index_to_mask = token_index
			if row[index_to_mask] not in dont_mask:
				row[index_to_mask] = mask_id
			
			token_index += 1
		
		return expanded_text, expanded, expanded_metadata
	
	expanded = dict(zip(inputs.keys(), ([] for _ in inputs.keys())))
	expanded_texts = []
	expanded_metadata = []
	for example in zip(*inputs.values(), dataset['test']['text'], metadata):
		ex_metadata = example[-1]
		text_example = example[-2]
		example = dict(zip(inputs.keys(), example[:-2]))
		text_expanded, ex_expanded, me_expanded = _expand_with_masks(example, text_example, ex_metadata, mask_id, dont_mask)
		for k in expanded:
			expanded[k].extend(ex_expanded[k])
		
		expanded_texts.extend(text_expanded)
		expanded_metadata.extend(me_expanded)
	
	for k in expanded:
		expanded[k] = torch.stack(expanded[k])
	
	expanded_dataset = DatasetDict(dict(test=Dataset.from_dict(dict(text=expanded_texts))))
	
	return expanded_dataset, expanded, expanded_metadata

def preprocess_dataset(
	dataset: Dataset,
	metadata: List[Dict], 
	data_args: DataArguments, 
	tokenizer: AutoTokenizer
) -> Tuple:
	'''
	Formats the dataset for use with a model.
	
	params:
		dataset (Dataset)			: a huggingface dataset. Must contain a "test" split,
									  with examples in the "text" column.
		data_args (DataArguments)	: the arguments containing information about the data.
					  				  see the DataArguments class for more details.
		tokenizer (AutoTokenizer)	: the tokenizer to use to prepare the examples for the model.
	
	returns:
		Text dataset 				: the text dataset. returned unchanged unless
									  the input is expanded for use with a masked language model
		Dataset 					: the dataset formatted for use with a model.
		Metadata 					: the metadata for the model. returned unchanged unless
									  the input is expanded for use with a masked language model
	'''
	drop_cols = dataset['test'].column_names
	
	def preprocess_function(examples: List[str]) -> Dict:
		'''Tokenizes a batch of string inputs.'''
		if not 'llama' in tokenizer.name_or_path or '-hf' in tokenizer.name_or_path:
			model_inputs = tokenizer(
				examples['text'], 
				max_length=data_args.max_source_length, 
				padding=True,
				truncation=True
			)
			
			start_ids = [tokenizer.bos_token_id, tokenizer.cls_token_id]
			start_ids = [token_id for token_id in start_ids if token_id is not None]
			if any(start_ids):
				start_id = start_ids[0]
				for i in range(len(model_inputs['input_ids'])):
					# add the cls/bos token to models that don't automatically include it
					# such as gpt2. we also need to ensure the other keys are the same length
					if model_inputs['input_ids'][i][0] != start_id:
						model_inputs['input_ids'][i].insert(0, start_id)
						for k in model_inputs.keys():
							if k == 'attention_mask':
								model_inputs[k][i].insert(0, 1)
							
							if k == 'token_type_ids':
								model_inputs[k][i].insert(0, 0)
		else:
			model_inputs = {
				'input_ids': [
					torch.tensor(tokenizer.encode(text, bos=True, eos=True)) for text in examples['text']
				]
			}
		
		return model_inputs
	
	test_dataset = dataset['test']
	
	if data_args.max_test_samples is not None:
		test_dataset = test_dataset.select(range(data_args.max_test_samples))
	
	test_dataset = test_dataset.map(
		preprocess_function,
		batched=True,
		num_proc=data_args.preprocessing_num_workers,
		remove_columns=drop_cols,
		load_from_cache_file=not data_args.overwrite_cache,
	)
	
	test_dataset.set_format(type='torch')
	
	if tokenizer.name_or_path in MASKED_LANGUAGE_MODELS | T5_MODELS:
		dont_mask = [t[0] for t in tokenizer(tokenizer.all_special_tokens, add_special_tokens=False)['input_ids']]
		
		# babyt5 handles commas and digits incorrectly; it tokenizes them as <unk>.
		# this is a very specific hack to get around this, without causing
		# other errors
		if 'babyt5' in tokenizer.name_or_path:
			dont_mask = [t for t in dont_mask if not t == tokenizer.unk_token_id]
		
		mask_id = tokenizer.mask_token_id
		# we don't use test_dataset.to_dict() because that converts it to a list of lists of ints,
		# and the expand_with_masks function takes Tensors
		inputs = {f: test_dataset[f] for f in test_dataset.features}
		dataset, inputs, metadata = expand_with_masks(
			inputs=inputs, 
			dataset=dataset, 
			metadata=metadata, 
			mask_id=mask_id, 
			dont_mask=dont_mask
		)
		test_dataset = Dataset.from_dict(inputs)
		test_dataset.set_format(type='torch')
	
	return dataset, test_dataset, metadata

def evaluate_model(
	model: Union[AutoModelForCausalLM, LLaMA],
	tokenizer: AutoTokenizer,
	examples: Dataset,
	metadata: List[Dict],
	test_dataset: Dataset,
	data_args: DataArguments
) -> None:
	'''
	Evaluates a model on the test dataset.
	Saves results in data_args.output_dir as a csv.
	
	params:
		model (Union[AutoModelForCausalLM, LLaMA]): the model to evaluate.
		tokenizer (AutoTokenizer)			: the tokenizer for the model
		examples (Dataset)					: the dataset as input strings.
		test_dataset (Dataset)				: the dataset to evaluate on
											  Should consist of sentences where
											  the position to be evaluated is 
											  replaced with constants.DEFAULT_MASK_TOKEN. Currently
											  evaluation of more than one replaced span is
											  not supported.
		metadata (List[Dict])				: the metadata for each example
		data_args (DataArguments)			: the arguments containing information about the data.
											  see the DataArguments class for more details.
	raises:
		ValueError 							: if eval_tokens for any sentence are not tokenized
											  as single tokens, predictions are hard to interpret,
											  so a ValueError is raised in this case.
	'''
	# this removes any slashes in the model name, which
	# causes problems when saving the results
	basename = re.sub(r'[\\/]', '-', model.name_or_path)
	output_pred_file = os.path.join(data_args.output_dir, basename + '.lm_results.csv.gz')
	
	# do not reevaluate if the output file already exists
	# currently disabled
	if os.path.exists(output_pred_file):
		# return
		pass
	
	def pad_tensor(t: torch.Tensor, pad: int, value: int = tokenizer.pad_token_id, dim: int = -1) -> torch.Tensor:
		'''
		Pads a tensor to length pad in dim dim.
		From https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8?u=mawilson
		
			params:
				t (torch.Tensor): tensor to pad
				pad (int)		: the size to pad to
				dim (int)		: dimension to pad
			
			returns:
				a new torch.Tensor padded to 'pad' in dimension 'dim'
		'''
		pad_size = list(t.shape)
		pad_size[dim] = pad - t.size(dim)
		return torch.cat([t, torch.full(size=pad_size, fill_value=value, dtype=t.dtype, device=t.device)], dim=dim)
	
	def pad_batch(batch: Tuple) -> Tuple:
		'''Pads examples in a batch to the same length.'''
		max_len = max(map(lambda ex: ex['input_ids'].size(-1), batch))
		batch = list(map(lambda ex: {k: pad_tensor(ex[k], pad=max_len, dim=-1) for k in ex}, batch))
		batch = {k: torch.stack([ex[k] for ex in batch], dim=0) for k in batch[0].keys()}
		return batch
	
	dataloader = DataLoader(
		test_dataset,
		batch_size=data_args.per_device_test_batch_size,
		collate_fn=pad_batch
	)
	
	if not 'llama' in model.name_or_path or '-hf' in model.name_or_path:
		model.eval()
	
	# we need to make the directory early
	# if we're saving tmp results there
	if data_args.save_tmp:
		os.makedirs(data_args.output_dir, exist_ok=True)
	
	n_observed_examples = 0
	metrics = []
	for i, inputs in tqdm(enumerate(dataloader), total = len(dataloader)):
		n_examples_in_batch = inputs['input_ids'].shape[0]
		
		# use this as a unique input identifier
		input_nums = range(n_observed_examples, n_observed_examples + n_examples_in_batch)
		n_observed_examples += n_examples_in_batch
		
		if data_args.save_tmp and os.path.isfile(f'{output_pred_file}_tmpbatch{i}.json.gz'):
			with gzip.open(f'{output_pred_file}_tmpbatch{i}.json.gz', 'rt') as in_file:
				metrics.extend([json.loads(l.strip()) for l in in_file.readlines()])
			
			continue
		
		input_texts = examples[(n_observed_examples - n_examples_in_batch):n_observed_examples]['text']
		batch_metadata = metadata[(n_observed_examples - n_examples_in_batch):n_observed_examples]
		
		results = evaluate_batch(
			model=model, 
			tokenizer=tokenizer,
			inputs=inputs, 
			input_texts=input_texts,
			input_nums=input_nums,
			batch_metadata=batch_metadata
		)
		
		if data_args.save_tmp:
			with gzip.open(f'{output_pred_file}_tmpbatch{i}.json.gz', 'wt') as out_file:
				for result in results:
					out_file.write(f'{json.dumps(result)}\n')
		
		metrics.extend(results)			
	
	if data_args.save_tmp:
		for file in glob(f'{output_pred_file}_tmpbatch*.json.gz'):
			os.remove(file)		
	
	metrics = pd.DataFrame(metrics)
	
	test_dataset_name = os.path.basename(data_args.test_file).replace('.txt.gz', '')
	
	metrics = metrics.assign(
		model_name=re.sub('["\']', '', model.config.name_or_path),
		task=get_model_task(model.config.name_or_path),
		n_params=f'{round(model.num_parameters()/1000000)}M',
		test_dataset=test_dataset_name,
		n_test_examples=test_dataset.num_rows,
	)
	
	move_to_beginning = ['model_name', 'task', 'n_params', 'test_dataset', 'n_test_examples']
	metrics = metrics[move_to_beginning + [c for c in metrics.columns if not c in move_to_beginning]]
	
	# we've already created the output directory
	# if we're saving tmp files
	if not data_args.save_tmp:
		os.makedirs(data_args.output_dir, exist_ok=True)
	
	metrics.to_csv(output_pred_file, index=False, na_rep='NA')

def get_model_task(model_name_or_path: str) -> str:
	'''Returns the model task based on the name.'''
	if model_name_or_path in NEXT_WORD_MODELS:
		return 'LM'
	
	if model_name_or_path in MASKED_LANGUAGE_MODELS:
		return 'MLM'
	
	if model_name_or_path in SEQ2SEQ_MODELS:
		return 'Seq2Seq'
	
	raise ValueError(f'{model_name_or_path!r} not found in `ALL_MODELS`!')

def evaluate_batch(
	model: Union[AutoModelForCausalLM, LLaMA],
	tokenizer: AutoTokenizer,
	inputs: Dict[str,torch.Tensor],
	input_texts: List[str],
	input_nums: List[int] = None,
	batch_metadata: List[Dict] = None,
) -> List[Dict]:
	'''
	Evaluates a batch of examples for a Language Model.
	For each input, determines the log probability of each eval token
	as a prediction for the next token.
	'''
	if input_nums is None:
		input_nums = range(len(inputs['input_ids'].shape[0]))
	
	if batch_metadata is None:
		batch_metadata = [{}] * inputs['input_ids'].shape[0]
	
	model_eval_function = get_model_eval_function(model_name_or_path=model.name_or_path)
	
	return model_eval_function(
		model=model,
		tokenizer=tokenizer,
		inputs=inputs,
		input_texts=input_texts,
		input_nums=input_nums,
		batch_metadata=batch_metadata
	)
	
def get_model_eval_function(model_name_or_path: str) -> Callable:
	'''
	Returns the appropriate function for eval based on the kind of 
	model.
	'''
	if model_name_or_path in NEXT_WORD_MODELS:
		return evaluate_LM_batch	
	
	if model_name_or_path in MASKED_LANGUAGE_MODELS:
		return evaluate_MLM_batch
	
	if model_name_or_path in T5_MODELS:
		return evaluate_T5_batch
	
	raise ValueError(model_not_supported_message(model_name_or_path))

def evaluate_LM_batch(
	model: Union[AutoModelForCausalLM, LLaMA],
	tokenizer: AutoTokenizer,
	inputs: Dict[str,torch.Tensor],
	input_texts: List[str],
	input_nums: List[int] = None,
	batch_metadata: List[Dict] = None,
) -> List[Dict]:
	'''
	Evaluates a batch of examples for a Language Model.
	For each input, determines the surprisal of each eval token
	as a prediction for the next token.
	'''
	with torch.no_grad():
		batch_outputs = model(**inputs)
	
	# convert to base 2 instead of base e
	batch_surprisals = -(1/torch.log(torch.tensor(2.))) * F.log_softmax(batch_outputs.logits, dim=-1)
	
	next_word_ids = tokenize_texts(tokenizer=tokenizer, text=input_texts)
	
	metrics = []
	records = zip(input_nums, input_texts, next_word_ids, batch_surprisals, batch_metadata)
	for input_num, input_text, next_word_tokens, surprisal, example_metadata in records:
		input_words = input_text.split()
		aligned_tokens = align_words_to_subword_tokens(
			tokenizer=tokenizer, 
			words=input_words, 
			tokens=next_word_tokens
		)
		
		tokens_seen = 0
		for word_num, tokens in enumerate(aligned_tokens):
			for token_num, token in enumerate(tokens):
				metrics.extend([{
					'item': input_num,
					'input_text': input_text,
					'word_num': word_num,
					'token_num_in_word': token_num,
					'token': tokenizer.decode(token),
					'token_id': token,
					'token_is_start_of_word': token_num == 0,
					'token_is_word': len(tokens) == 1,
					'surprisal': surprisal[tokens_seen,token].item(),
					'predicted_token': tokenizer.decode(
						torch.argmin(surprisal[tokens_seen,:], dim=-1).item()
					),
					**example_metadata,
				}])
				tokens_seen += 1
	
	return metrics

def align_words_to_subword_tokens(
	tokenizer: AutoTokenizer, 
	words: List[str], 
	tokens: List[int]
) -> List[List[int]]:
	'''
	Aligns words to subword tokens. Note that this does not currently
	work for uncased models.
	
	params:
		tokenizer: AutoTokenizer: the tokenizer used to generate `tokens`
		words (List[str]): a list of words to align
		tokens (List[int]): a list of tokens generated from the sequence of words.
	
	returns:
		List of list of ints, which is the same length as `words`.
		Each sublist contains the tokens corresponding to the word
		at the same position as the sublist in `words`.	
	
	raises:
		IndexError: if the words and tokens cannot be aligned.
	'''
	# pop works backward
	num_words = len(words)
	tokens = tokens[::-1]
	words = words[::-1]
	
	# handle uncased and cased tokenizers
	tokenizer_kwargs = dict()
	if 'llama' in tokenizer.name_or_path and not '-hf' in tokenizer.name_or_path:
		tokenizer_kwargs = dict(bos=False, eos=False)
	
	uncased = tokenizer.tokenize('A', **tokenizer_kwargs) == tokenizer.tokenize('a', **tokenizer_kwargs)
	if uncased:
		words = [w.lower() for w in words]
	
	aligned = []
	while tokens:
		aligned_tokens = [tokens.pop()]
		word = words.pop()
		
		# we need to replace all spaces here rather than
		# just stripping because some tokenizer don't handle
		# words with punctuation in the middle correctly
		# e.g, 'bert-large-cased' tokenizes 're-wrapped' as
		# [1231, 118, 4293], but decodes that sequence as
		# 're - wrapped', with spaces in the middle.
		if 'babyt5' in tokenizer.name_or_path:
			# babyt5 doesn't tokenize commas correctly, but
			# as its <unk> token. in general, an <unk> token
			# should not be used to identify a word, since
			# not all <unk> tokens have the same source.
			# in this case, we build in a very specific hack.
			# we don't want a more general solution, since
			# that could mask an actually problematic case
			while re.sub(r'\s', '', tokenizer.decode(aligned_tokens)) != re.sub('[0-9,]', tokenizer.unk_token, word):
				aligned_tokens += [tokens.pop()]
		else:
			while re.sub(r'\s', '', tokenizer.decode(aligned_tokens)) != word:
				aligned_tokens += [tokens.pop()]
	
		aligned.append(aligned_tokens)
	
	assert len(aligned) == num_words, (
		f'Unable to find {num_words} in text.'
	)
	
	return aligned

def evaluate_MLM_batch(
	model: AutoModelForMaskedLM,
	tokenizer: AutoTokenizer,
	inputs: Dict[str,torch.Tensor],
	input_texts: List[str],
	input_nums: List[int] = None,
	batch_metadata: List[Dict] = None,
) -> List[Dict]:
	'''
	Evaluates a batch of examples for a Language Model.
	For each input, determines the surprisal of token
	corresponding to the masked token.
	'''
	if batch_metadata is None:
		batch_metadata = [{}] * inputs['input_ids'].shape[0]
	
	with torch.no_grad():
		batch_outputs = model(**inputs)
	
	# these are the positions we want to extract predictions from
	mask_locations = torch.nonzero(inputs['input_ids'] == tokenizer.mask_token_id, as_tuple=True)[-1]
	
	# convert to base 2 instead of base e
	batch_surprisals = -(1/torch.log(torch.tensor(2.))) * F.log_softmax(batch_outputs.logits, dim=-1)
	
	word_ids = tokenize_texts(tokenizer=tokenizer, text=input_texts)
	
	special_token_ids = [t[0] for t in tokenizer(tokenizer.all_special_tokens, add_special_tokens=False)['input_ids'] if not t[0] == tokenizer.mask_token_id]
	starting_ids = [[t not in special_token_ids for t in row].index(True) for row in inputs['input_ids']]
	
	metrics = []
	records = zip(input_nums, input_texts, inputs['input_ids'], starting_ids, mask_locations, word_ids, batch_surprisals, batch_metadata)
	for input_num, input_text, input_ids, starting_id, mask_location, word_tokens, surprisal, example_metadata in records:
		input_words = input_text.split()
		aligned_tokens = align_words_to_subword_tokens(
			tokenizer=tokenizer, 
			words=input_words, 
			tokens=word_tokens
		)
		
		# we need to extract the aligned token set that has the token_num in it,
		# so we can determine whether it starts a word or not
		
		# this tells us which word contains the token in the masked location
		starting_token_numbers = []
		for i, _ in enumerate(aligned_tokens):
			starting_token_numbers.append(sum(len(t) for t in aligned_tokens[:i]))
		
		starting_token_num_of_word = mask_location - starting_id
		while starting_token_num_of_word not in starting_token_numbers:
			starting_token_num_of_word -= 1
		
		# then, we get the tokens for that word,
		# find the token's position in that word (the location of the mask token),
		# and extract the actual token we want the prediction for from the word
		tokens = aligned_tokens[starting_token_numbers.index(starting_token_num_of_word)]
		token_num_in_word = input_ids[starting_token_num_of_word+starting_id:].tolist().index(tokenizer.mask_token_id)
		token = tokens[token_num_in_word]
		word_num = starting_token_numbers.index(starting_token_num_of_word.item())
		
		metrics.extend([{
			'item': input_num,
			'input_text': input_text,
			'word_num': word_num,
			'token_num_in_word': token_num_in_word,
			'token': tokenizer.decode(token),
			'token_id': token,
			'token_is_start_of_word': token_num_in_word == 0,
			'token_is_word': len(tokens) == 1,
			'surprisal': surprisal[mask_location,token].item(),
			'predicted_token': tokenizer.decode(
				torch.argmin(surprisal[mask_location,:], dim=-1).item()
			),
			**example_metadata,
		}])
	
	return metrics

def evaluate_T5_batch(
	model: AutoModelForMaskedLM,
	tokenizer: AutoTokenizer,
	inputs: Dict[str,torch.Tensor],
	input_texts: List[str],
	input_nums: List[int] = None,
	batch_metadata: List[Dict] = None,
) -> List[Dict]:
	'''
	Evaluates a batch of examples for a Language Model.
	For each input, determines the surprisal of token
	corresponding to the masked token.
	'''
	def prefix_allowed_tokens_fn_factory(tokenizer: AutoTokenizer) -> Callable[[int,torch.Tensor], List[int]]:
		'''
		Returns a function that constrains the output generation to start
		with the mask span token.
		'''
		pad_token_id = tokenizer.pad_token_id
		mask_span_id = tokenizer.mask_token_id
				
		def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
			'''
			Determines which tokens can be predicted at each decoding step, according to the 
			metadata and general constraints (i.e., mask span token must be first).
			'''
			if len(input_ids) == 0:
				# first token must be the pad token
				return [pad_token_id]
			elif len(input_ids) == 1:
				# second token must be the mask span token
				return [mask_span_id]
			else:
				# subsequent tokens can be anything,
				# and we will extract the log probabilities
				# for the eval tokens later
				return list(range(tokenizer.vocab_size))
		
		return prefix_allowed_tokens_fn
	
	if batch_metadata is None:
		batch_metadata = [{}] * inputs['input_ids'].shape[0]
	
	with torch.no_grad():
		batch_outputs = model.generate(
			inputs['input_ids'],
			prefix_allowed_tokens_fn=prefix_allowed_tokens_fn_factory(
				tokenizer=tokenizer
			),
			return_dict_in_generate=True,
			max_new_tokens=2,
			output_scores=True
		)
	
	# these are the positions we want to extract predictions from
	# we need them here not to get the surprisal (since that is
	# always at the final index), but to figure out which word
	# the token belongs to
	mask_locations = torch.nonzero(inputs['input_ids'] == tokenizer.mask_token_id, as_tuple=True)[-1]
	
	# convert to base 2 instead of base e
	batch_surprisals = -(1/torch.log(torch.tensor(2.))) * F.log_softmax(batch_outputs['scores'][-1], dim=-1)
	
	word_ids = tokenize_texts(tokenizer=tokenizer, text=input_texts)
	
	special_token_ids = [t[0] for t in tokenizer(tokenizer.all_special_tokens, add_special_tokens=False)['input_ids'] if not t[0] == tokenizer.mask_token_id]
	starting_ids = [[t not in special_token_ids for t in row].index(True) for row in inputs['input_ids']]
	
	metrics = []
	records = zip(input_nums, input_texts, inputs['input_ids'], starting_ids, mask_locations, word_ids, batch_surprisals, batch_metadata)
	for input_num, input_text, input_ids, starting_id, mask_location, word_tokens, surprisal, example_metadata in records:
		input_words = input_text.split()
		aligned_tokens = align_words_to_subword_tokens(
			tokenizer=tokenizer, 
			words=input_words, 
			tokens=word_tokens
		)
		
		# we need to extract the aligned token set that has the token_num in it,
		# so we can determine whether it starts a word or not
		
		# this tells us which word contains the token in the masked location
		starting_token_numbers = []
		for i, _ in enumerate(aligned_tokens):
			starting_token_numbers.append(sum(len(t) for t in aligned_tokens[:i]))
		
		starting_token_num_of_word = mask_location - starting_id
		while starting_token_num_of_word not in starting_token_numbers:
			starting_token_num_of_word -= 1
		
		# then, we get the tokens for that word,
		# find the token's position in that word (the location of the mask token),
		# and extract the actual token we want the prediction for from the word
		tokens = aligned_tokens[starting_token_numbers.index(starting_token_num_of_word)]
		token_num_in_word = input_ids[starting_token_num_of_word+starting_id:].tolist().index(tokenizer.mask_token_id)
		token = tokens[token_num_in_word]
		word_num = starting_token_numbers.index(starting_token_num_of_word.item())
		
		metrics.extend([{
			'item': input_num,
			'input_text': input_text,
			'word_num': word_num,
			'token_num_in_word': token_num_in_word,
			'token': tokenizer.decode(token),
			'token_id': token,
			'token_is_start_of_word': token_num_in_word == 0,
			'token_is_word': len(tokens) == 1,
			'surprisal': surprisal[token].item(),
			'predicted_token': tokenizer.decode(
				torch.argmin(surprisal, dim=-1).item()
			),
			**example_metadata,
		}])
	
	return metrics

def tokenize_texts(tokenizer: AutoTokenizer, text: List[str]) -> List[List[int]]:
	'''
	Tokenize a list of examples without special tokens for use during evaluation.
	'''
	if not 'llama' in tokenizer.name_or_path or '-hf' in tokenizer.name_or_path:
		tokenized = tokenizer(text, add_special_tokens=False)['input_ids']
	else:
		tokenized = [tokenizer.encode(ex, bos=False, eos=False) for ex in text]
	
	return tokenized

def load_metadata(dataset_path: str) -> List[Dict]:
	'''
	Loads the metadata file for a dataset.
	'''
	with gzip.open(dataset_path.replace('.txt.gz', '_metadata.json.gz'), 'rt', encoding='utf-8') as in_file:
		metadata = [json.loads(l) for l in in_file.readlines()]	
	
	return metadata

def extract_surprisals() -> None:
	'''Main function.'''
	model_args, data_args = parse_cl_arguments(ModelArguments, DataArguments)
	
	logger.info(f'Model parameters: {model_args}')
	logger.info(f'Evaluation parameters: {data_args}')
	
	dataset = load_dataset('text', data_files={'test': data_args.test_file})
	metadata = load_metadata(data_args.test_file)
	if not 'llama' in model_args.model_name_or_path or '-hf' in model_args.model_name_or_path:
		tokenizer, model = load_HF_tokenizer_and_model(model_args)
		dataset, test_dataset, metadata = preprocess_dataset(dataset=dataset, metadata=metadata, data_args=data_args, tokenizer=tokenizer)
	else:
		tokenizer = load_llama_tokenizer(tokenizer_path=model_args.tokenizer_name)
		dataset, test_dataset, metadata = preprocess_dataset(dataset=dataset, metadata=metadata, data_args=data_args, tokenizer=tokenizer)
		max_seq_len = max(len(ex) for ex in test_dataset['input_ids'])
		model = load_llama(
			ckpt_dir=model_args.model_name_or_path, 
			tokenizer=tokenizer, 
			max_seq_len=max_seq_len,
			max_batch_size=data_args.per_device_test_batch_size
		)
	
	evaluate_model(
		model=model, 
		tokenizer=tokenizer, 
		examples=dataset['test'],
		metadata=metadata,
		test_dataset=test_dataset, 
		data_args=data_args
	)

if __name__ == '__main__':
	extract_surprisals()
