from typing import Dict, List

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer


class MyDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: BertTokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        text = self.data[index]["text"]
        label = self.data[index]["label"]

        text_encoded = self.tokenizer(text=text,
                                      add_special_tokens=True,  # 开头加 [CLS], 末尾加 [SEP]
                                      truncation=True,
                                      max_length=256,
                                      return_tensors='pt')

        return {
            "input_ids": text_encoded["input_ids"],
            "attention_mask": text_encoded["attention_mask"],
            "label": torch.tensor(label)
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids_list = [instance['input_ids'][0] for instance in batch]
        input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        attention_mask_list = [instance['attention_mask'][0] for instance in batch]
        attention_mask_pad = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        label_list = [instance['label'] for instance in batch]

        return {
            "input_ids": torch.tensor(input_ids_pad),
            "attention_mask": torch.tensor(attention_mask_pad),
            "label": torch.tensor(label_list)
        }
