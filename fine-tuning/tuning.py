import os
import argparse
import dotenv
from datasets import Dataset


import torch
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from collator import Collator
from training_callback import TokenSpeedCallback, early_stop

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

GEMMA_DATASET_ROOT_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'gemma-dataset')
PHI_DATASET_ROOT_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'phi-dataset')


SEED = 42
torch.manual_seed(SEED)

def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def setup_wandb_env(
    wandb_project: str = None,
    wandb_run_name: str = None,
    wandb_mode: str = "online",
):
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ.setdefault("WANDB_WATCH", "false")
        os.environ["WANDB_MODE"] = wandb_mode
        os.environ["WANDB_DISABLE_CODE"] = "true"
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        if wandb_run_name:
            os.environ["WANDB_NAME"] = wandb_run_name
        print(f"[W&B] Enabled. Project={wandb_project}, Run={wandb_run_name}, Mode={wandb_mode}")
    else:
        os.environ["WANDB_DISABLED"] = "true"
        print("[W&B] Disabled (no --wandb_project provided).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma 3 with Unsloth + QLoRA on pre-tokenized data, with W&B logging."
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing subfolders of JSONL files with pre-tokenized input_ids.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length to keep from pre-tokenized sequences.",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.01,
        help="Fraction of data to use for eval (0.0 disables eval).",
    )

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help=(
            "Base model name to load via Unsloth FastLanguageModel. "
            "Example: 'unsloth/gemma-3-4b-bnb-4bit' or another Gemma 3 checkpoint."
        ),
    )

    # Training
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Output directory for checkpoints.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Base learning rate for LoRA weights & other trainable params.",
    )
    parser.add_argument(
        "--embedding_learning_rate", # <- unsloth specific arg
        type=float,
        default=5e-6,
        help="Smaller lr for embeddings / lm_head (2-10x smaller than LR).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging interval (steps).",
    )

    # LoRA
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--use_rslora",
        action="store_true",
        default=True,
        help="Enable RSLoRA in Unsloth.",
    )

    # W&B
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name. If set, W&B logging is enabled.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional W&B run name.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline"],
        help="W&B mode: 'online' or 'offline'.",
    )
    parser.add_argument(
        "--hf_path",
        type=str,
        default=None,
        help="W&B mode: 'online' or 'offline'.",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint folder (e.g. models/checkpoint-1500) to resume training from.",
    )

    return parser.parse_args()

def create_trainer(
    model,
    tokenizer,
    max_seq_length,
    train_dataset,
    eval_dataset,
    data_collator,
    output_dir,
    learning_rate,
    embedding_learning_rate,
    batch_size,
    gradient_accumulation_steps,
    num_train_epochs,
    logging_steps,
    wandb_project,
    wandb_run_name,
):

    if wandb_project:
        report_to = ["wandb"]
    else:
        report_to = ["none"]

    speed_cb = TokenSpeedCallback(seq_len=max_seq_length, batch_size=batch_size, grad_accum=gradient_accumulation_steps)

    training_args = UnslothTrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        embedding_learning_rate=embedding_learning_rate,
        warmup_ratio=0.05,

        optim="adamw_8bit",
        weight_decay=0.01,

        logging_steps=logging_steps,
        report_to=report_to,
        run_name=wandb_run_name,

        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,

        eval_strategy="steps",
        eval_steps=100,

        output_dir=output_dir,
        seed=SEED,
        lr_scheduler_type='cosine',

        load_best_model_at_end = True,       # MUST USE for early stopping
        metric_for_best_model = "eval_loss", # metric we want to early stop on

        remove_unused_columns = False
    )

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        args=training_args,
        callbacks=[speed_cb, early_stop]
    )
    return trainer

def train(trainer, resume_from_checkpoint):
    print("[TRAIN] Starting training...")
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("[TRAIN] Done.")

    metrics = getattr(trainer_stats, "metrics", None) or {}
    if "train_runtime" in metrics:
        minutes = round(metrics["train_runtime"] / 60.0, 2)
        print(f"[TRAIN] Train runtime: {minutes} minutes")
    if "train_loss" in metrics:
        print(f"[TRAIN] Final train loss: {metrics['train_loss']:.4f}")

    return trainer_stats


def main():
    dotenv.load_dotenv('.env')
    args = parse_args()


    setup_wandb_env(
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode
    )
    

    print(f"Loading dataset from {args.data_dir}...")
    dataset = Dataset.load_from_disk(dataset_path=args.data_dir)
    print(f"Total samples: {len(dataset)}")

    split = dataset.train_test_split(test_size=args.eval_ratio, seed=SEED, shuffle=True)
    train_ds = split['train']
    test_ds = split['test']

    print(f"Training samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")

    print(f"Loading model {args.model_name} via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True # <- qLoRA 4-bit quantization
    )
    print("Preparing LoRA adapters (qLoRA) ...")
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=args.use_rslora,
    )

    # collator
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    print(f'pad_id={pad_id}')   
    collator = Collator(pad_token_id=pad_id, max_length=args.max_seq_length)

    device = get_device()
    print(f"Using {device}")

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        embedding_learning_rate=args.embedding_learning_rate * 1e-1,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    train(trainer, resume_from_checkpoint=args.resume_from_checkpoint)

    print("Saving LoRA adapters + tokenizer...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir + '/model')
    tokenizer.save_pretrained(args.output_dir + '/model')
    
    # tokenizer.save_pretrained(args.output_dir)
    # print(f"[SAVE] Done. Artifacts in: {args.output_dir / 'model'}")
    print('[UPLOAD] Uploading to hub...')
    model.push_to_hub(
        args.hf_path,
        token=os.getenv("HF_TOKEN"),
    )
    tokenizer.push_to_hub(
        args.hf_path,
        token=os.getenv("HF_TOKEN"),
    )
    print('[UPLOAD] Done')

if __name__ == "__main__":
    main()
    

