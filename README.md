# extract-surprisals

A simple interface for extracting surprisals for each word position from autoregressive language models.

## Detailed description

This repo provides code to make extracting single-token predictions from autoregressive language models.

First, we will discuss the structure of the datasets, since this will make it easier to discuss how evaluation occurs.

### Dataset structure

Datasets go in the `data` directory. Each subdirectory contains a single dataset, which should bear the same name as the subdirectory it occurs in. A dataset consists of two files, a `txt.gz` file, and a `_metadata.json.gz` file.

#### `$dataset.txt.gz`

The `$dataset.txt.gz` file contains a single example per line.

```
I want to evaluate this sentence.
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

The models are run autoregressively, and for each input position, produce a probability distribution of the next token position given the input up to that position. For each position, the surprisal of the actual next token is extracted.

### Setup

Use `conda/mamba env create -f environment.yaml` to replicate our conda environment.