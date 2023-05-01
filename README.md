# extract-surprisals

A simple interface for extracting surprisals for each word position from autoregressive language models.

## Detailed description

This repo provides code to make extracting token-by-token predictions for target sentences from auto-regressive language models, masked language models, and conditional generation models (= T5 models) easy.

First, we will discuss the structure of the datasets, since this will make it easier to discuss how evaluation occurs.

### Dataset structure

Datasets go in the `data` directory. Each subdirectory contains a single dataset, which should bear the same name as the subdirectory it occurs in. A dataset consists of two files, a `txt.gz` file, and a `_metadata.json.gz` file.

#### `$dataset.txt.gz`

The `$dataset.txt.gz` file contains a single example per line.

```
I want to evaluate this sentence, word-by-word.
I also would like to evaluate this sentence.
...
```

#### `$dataset_metadata.json.gz`

The `$dataset_metadata.json.gz` file consists of one json object per line, one for each example in `$dataset.txt.gz`. This should be a dictionary. If you do not want to record any metadata for a sentence, this file is not required (though it is highly encouraged).
```
{"condition": "a"}
{"condition": "b"}
...
```
You should include any key-value pairs you would like to record additional information about each item in the results (this is intended to make comparing different conditions of items convenient). Each row should contain all of the same keys (e.g., you should not include a key in row 2 that does not exist in row 1).

### Running the script

To run evaluation on a dataset, you can do the following:
```
python core/extract_surprisals.py --model_name_or_path ... --test_file ...
```
This will run the test file provided on the model. Results are saved in a `csv.gz` file in `output_dir` with a single row for each eval token for each sentence. Each row reports the input, various information about the model, the metadata from `$dataset_metadata.json.gz`, and the log probability of each eval token for each (corresponding) sentence.

You can also specify the following optional arguments:

- `config_name`: the name of the HF config to use (if different from the model name).
- `cache_dir`: where to store the pretrained models downloaded from HF.
- `tokenizer_name`: the name of the tokenizer to use (if different from the model name).
- `use_fast_tokenizer`: whether to use one of HF's fast tokenizers. Default: `True`, except for certain models. For OPT models, this is always set to `False` internally, since they do not support HF's fast tokenizer.
- `model_revision`: which model version to use (branch name, tag name, or commit id). Default: `main`.
- `use_auth_token`: where to use an auth token when trying to load a model. Required for use with private models. Default: `False`.

- `output_dir`: where to save the results. Default: `outputs/$dataset_name/$model_name.lm_results.csv.gz`.
- `overwrite_cache`: whether to overwrite the cached test dataset. Default: `False`.
- `preprocessing_num_workers`: the number of workers to use for preprocessing the dataset. Default: `None`.
- `max_source_length`: the maximum allowable sentence length after tokenization. Default: `1024`.
- `per_device_test_batch_size`: the batch size for the test dataset.
- `max_test_samples`: the number of examples to truncate the test dataset to. Intended for use when debugging. Default: `None` (i.e., all examples included).
- `save_tmp`: whether to save temporary gzipped json files with the results of each batch during evaluation. This is useful if you are unable to finish evaluation on your full test dataset in one go (e.g., if you are running a large model on a cluster and cannot finish within job time limits). If these temp files exist, results will be loaded from them instead of by running the model, allowing you to skip over batches that have already been evaluated. Once all batches are evaluated on a run, the temp files will be deleted.

### Under the hood

Auto-regressive models represent the most straightforward case. Such models are run autoregressively, and for each input position, produce a probability distribution of the next token position given the input up to that position. For each position, the surprisal of the actual next token from the input is extracted.

For masked language models and conditional generation models, each input sentence is tokenized, and then repeated with each non-special token replaced with a mask or mask span token.  Without masking, the models would have a much higher probability of predicting the actual word, so masking makes the evaluation more meaningful: how surprising is the actual word when it is not given? The metadata is also automatically repeated the correct number of times so it lines up with the repeated inputs. Then, the models are run and surprisal values are extracted for each position when it is masked in each sentence for the token that goes there in the actual sentence.

For conditional generation models, we additionally use teacher-forcing to force the first output token to be the BOS token and the second output token to be the mask token, before extracting the surprisal value from the final token.

### Extensions

If you want to add a new model that falls into one of these bins, it's pretty easy. If its tokenizer has no special requirements, you can just add the model in the correct set in `core/constants.py`. If you want to run it as a masked language model, it should be added to `MASKED_LANGUAGE_MODELS`; to run it as an autoregressive language model, add it to `NEXT_WORD_MODELS`; and to run it on a conditional generation task, you should add it to `T5_MODELS`.

If there are special things about the model's tokenizer, you should modify the `get_tokenizer_kwargs` function in `core/constants.py` (e.g., the BabyBERTa model tokenizers require `add_prefix_space=True`).

### Setup

Use `conda/mamba env create -f environment.yaml` to replicate our conda environment.