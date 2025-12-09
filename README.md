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

# Run fine-tuning process
Model: unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit and unsloth/gemma-3-4b-it-unsloth-bnb-4bit
```bash
conda run python fine-tuning/tuning.py \ 
--data_dir dataset/gemma-dataset/ \
--model_name unsloth/gemma-3-4b-it-unsloth-bnb-4bit \
--output_dir models/gemma3-4b-qlora \
--num_train_epochs 1 \
--learning_rate 1e-5 \
--embedding_learning_rate 1e-6 \
--wandb_project scc450-slm-llm \
--wandb_run_nam gemma3-4b-qlora-run-1
```

conda run python fine-tuning/tuning.py --data_dir dataset/gemma-dataset/ --model_name unsloth/gemma-3-4b-it-unsloth-bnb-4bit --output_dir models/gemma3-4b-qlora --num_train_epochs 1 --learning_rate 1e-5 --embedding_learning_rate 1e-6 --wandb_project scc450-slm-llm --wandb_run_name gemma3-4b-qlora-run-1

conda run python fine-tuning/tuning.py --data_dir dataset/phi-dataset/ --model_name unsloth/Phi-3-mini-4k-instruct-bnb-4bit --output_dir models/phi3-mini-qlora --num_train_epochs 1 --learning_rate 1e-5 --embedding_learning_rate 1e-6 --wandb_project scc450-slm-llm --wandb_run_name phi-3-mini-instruct-qlora-run-1 --hf_path nhannguyen2730/phi-3-mini-instruct-qlora-tc