import os
import json
from datasets import Dataset

# Get the script directory to build absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

GEMMA_TOKENIZED_ROOT_DIR = os.path.join(PROJECT_ROOT, 'gemma_tokenized_trafford', 'tokenized')
PHI_TOKENIZED_ROOT_DIR = os.path.join(PROJECT_ROOT, 'phi_tokenized_trafford', 'tokenized')

def walk_files(root):
    subdirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

    # 15 last files are for evaluation
    subdirs_to_process = subdirs[:-15] if len(subdirs) > 15 else []

    for subdir in subdirs_to_process:
        subdir_path = os.path.join(root, subdir)
        for dirpath, _, filenames in os.walk(subdir_path):
            for f in filenames:
                if f.endswith('.json'):
                    yield os.path.join(dirpath, f)

def make_dataset(root, max_seq_len, output):
    def gen():
        for path in walk_files(root):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    ids = data['input_ids']

                    ids = ids[:max_seq_len]

                    yield {
                        "input_ids": ids,
                        "attention_mask": [1] * len(ids),
                    }

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = Dataset.from_generator(gen)
    dataset.save_to_disk(output)

make_dataset(GEMMA_TOKENIZED_ROOT_DIR, 2048, "dataset/gemma-dataset")
make_dataset(PHI_TOKENIZED_ROOT_DIR, 2048, "dataset/phi-dataset")

