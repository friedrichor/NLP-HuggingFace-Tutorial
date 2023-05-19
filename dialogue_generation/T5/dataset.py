from typing import Dict, List

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer


class MyDataset(Dataset):
    def __init__(self, data: List[Dict], text_prefix: str, tokenizer: T5Tokenizer):
        self.data = data
        self.text_prefix = text_prefix
        self.tokenizer = tokenizer

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        context = self.data[index]["context"]
        response = self.data[index]["response"]

        context_encoded = self.tokenizer(text=self.text_prefix + context,
                                         add_special_tokens=True,
                                         # padding="max_length",
                                         truncation=True,
                                         max_length=256,
                                         return_tensors='pt')

        response_encoded = self.tokenizer(text=response,
                                          add_special_tokens=True,
                                          # padding="max_length",
                                          truncation=True,
                                          max_length=128,
                                          return_tensors='pt')
        labels = response_encoded["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": context_encoded["input_ids"],
            "attention_mask": context_encoded["attention_mask"],
            "labels": labels
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids_list = [instance["input_ids"][0] for instance in batch]
        input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        attention_mask_list = [instance["attention_mask"][0] for instance in batch]
        attention_mask_pad = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        labels_list = [instance["labels"][0] for instance in batch]
        labels_pad = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        return {
            "input_ids": torch.tensor(input_ids_pad),
            "attention_mask": torch.tensor(attention_mask_pad),
            "labels": torch.tensor(labels_pad)
        }
