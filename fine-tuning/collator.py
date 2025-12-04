import torch
from dataclasses import dataclass
@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attn_masks = []
        labels = []

        for item in batch:
            ids = item['input_ids']
            attn = item['attention_mask']
            label = item['labels']

            pad_len = max_len - len(ids)

            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attn_masks.append(attn + [0] * pad_len)
            labels.append(label + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, torch.long),
            "attention_mask": torch.tensor(attn_masks, torch.long),
            "labels": torch.tensor(labels, torch.long)
        }
