import torch
from dataclasses import dataclass
@dataclass
class Collator:
    pad_token_id: int
    max_length: int

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        input_ids = []
        attn_masks = []
        labels = []

        for item in batch:
            print('='* 30)
            print(item)
            print('='* 30)

            ids = item['input_ids'][:max_len]
            attn = item['attention_mask'][:max_len]

            label = ids

            pad_len = max_len - len(ids)

            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attn_masks.append(attn + [0] * pad_len)
            labels.append(label + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, torch.long),
            "attention_mask": torch.tensor(attn_masks, torch.long),
            "labels": torch.tensor(labels, torch.long)
        }
