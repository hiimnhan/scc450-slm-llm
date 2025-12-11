# Installation

1. Create and activate environment

- Create a new environment from `environment.yaml` file.

```bash
conda env create -f environment.yaml
```

- Activate

```bash
conda activate scc450-slm-llm
```

# Data Extraction

```bash
conda run python convert_docs_v3.py \
-i <input folder> \
-s <strategy used to extract, ['hi_res', 'vlm'], default is hi_res> \
-o <out_put directory> \
-p <number of processes you want to use> \
--use-api <True/False, come along with strategy vlm>
```

# Data conversion from extracted data

```bash
conda run python extracted_structured_data.py
```

# Run fine-tuning process

_Prerequisites_
A computer run Linux with NVIDA GPU card

If you use wandb for monitoring and logging, first run

```bash
wandb login
```

Our implementation includes pushing models to Hugging Face space, so please add `HF_TOKEN` in `.env` file first.

```bash
conda run python fine-tuning/tuning.py \
--data_dir \
--model_name \
--output_dir \
--num_train_epochs 1 \
--learning_rate 1e-5 \
--embedding_learning_rate 1e-6 \
--wandb_project project_name \
--wandb_run_name run_name \
--hf_path hf_path/model-xxx
```

Example

- Model unsloth/gemma-3-4b-it-unsloth-bnb-4bit

````bash
conda run python fine-tuning/tuning.py --data_dir dataset/gemma-dataset/ --model_name unsloth/gemma-3-4b-it-unsloth-bnb-4bit --output_dir models/gemma3-4b-qlora --num_train_epochs 1 --learning_rate 1e-5 --embedding_learning_rate 1e-6 --wandb_project scc450-slm-llm --wandb_run_name gemma3-4b-qlora-run-1
``

- Model unsloth/Phi-3-mini-4k-instruct-bnb-4bit

```bash
conda run python fine-tuning/tuning.py --data_dir dataset/phi-dataset/ --model_name unsloth/Phi-3-mini-4k-instruct-bnb-4bit --output_dir models/phi3-mini-qlora --num_train_epochs 1 --learning_rate 1e-5 --embedding_learning_rate 1e-6 --wandb_project scc450-slm-llm --wandb_run_name phi-3-mini-instruct-qlora-run-1 --hf_path nhannguyen2730/phi-3-mini-instruct-qlora-tc
````

## Tunable Parameters

This section describes the key parameters you can adjust to configure the fine-tuning process.

### Data Parameters

- `--data_dir` (required):
  **Type**: `str`
  The root directory containing subfolders of JSONL files with pre-tokenized `input_ids`.

- `--max_seq_length`:
  **Type**: `int` (default: `1024`)
  The maximum sequence length to keep from pre-tokenized sequences.

- `--eval_ratio`:
  **Type**: `float` (default: `0.01`)
  The fraction of the dataset to use for evaluation. Setting this to `0.0` disables evaluation.

### Model Parameters

- `--model_name` (required):
  **Type**: `str`
  The base model name to load via Unsloth's FastLanguageModel. Example: `'unsloth/gemma-3-4b-bnb-4bit'`.

### Training Parameters

- `--output_dir`:
  **Type**: `str` (default: `models`)
  The directory where model checkpoints will be saved.

- `--learning_rate`:
  **Type**: `float` (default: `5e-5`)
  The base learning rate for LoRA weights and other trainable parameters.

- `--embedding_learning_rate`:
  **Type**: `float` (default: `5e-6`)
  A smaller learning rate for embeddings and the language model head (typically 2-10x smaller than the main learning rate).

- `--batch_size`:
  **Type**: `int` (default: `1`)
  The per-device batch size for training.

- `--gradient_accumulation_steps`:
  **Type**: `int` (default: `16`)
  The number of gradient accumulation steps to perform before updating the model weights.

- `--num_train_epochs`:
  **Type**: `float` (default: `1.0`)
  The number of training epochs to run.

- `--logging_steps`:
  **Type**: `int` (default: `10`)
  The frequency of logging during training, in terms of steps.

### LoRA (Low-Rank Adaptation) Parameters

- `--lora_r`:
  **Type**: `int` (default: `16`)
  The rank of the LoRA layers.

- `--lora_alpha`:
  **Type**: `int` (default: `32`)
  The scaling factor for LoRA layers.

- `--use_rslora`:
  **Type**: `bool` (default: `True`)
  Enables RSLoRA in Unsloth if set to `True`.

### W&B (Weights & Biases) Logging Parameters

- `--wandb_project`:
  **Type**: `str` (default: `None`)
  The name of the W&B project. If set, W&B logging is enabled.

- `--wandb_run_name`:
  **Type**: `str` (default: `None`)
  Optional custom run name for W&B logging.

- `--wandb_mode`:
  **Type**: `str` (default: `online`)
  The W&B mode. Choose between:
- `online`: For real-time logging.
- `offline`: For local logging without real-time updates.

- `--hf_path`:
  **Type**: `str` (default: `None`)
  Hugging Face path to push model.
  \*\*
