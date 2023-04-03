from typing import List, Dict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer


class MyDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: T5Tokenizer, classes_map: Dict, prefix_text: str):
        self.data = data
        self.tokenizer = tokenizer

        self.labels_map = {value: key for key, value in classes_map.items()}  # labels_map = {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}
        self.prefix_text = prefix_text

    def __getitem__(self, index):
        text = self.data[index]["text"]
        label = self.data[index]["label"]

        text_encoded = self.tokenizer.encode_plus(text=self.prefix_text + text,
                                                  add_special_tokens=True,  # T5没有[CLS]和[SEP]，有结束符</s>
                                                  return_attention_mask=True)
        labels_id = self.tokenizer.encode(text=self.labels_map[label],
                                          add_special_tokens=True)  # 在末尾添加 </s>

        return {
            "input_ids": text_encoded["input_ids"],
            "attention_mask": text_encoded["attention_mask"],
            "labels": labels_id
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids_list = [torch.tensor(instance['input_ids']) for instance in batch]
        input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        attention_mask_list = [torch.tensor(instance['attention_mask']) for instance in batch]
        attention_mask_pad = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        labels_list = [torch.tensor(instance['labels']) for instance in batch]
        labels_pad = pad_sequence(labels_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        return {
            "input_ids": torch.tensor(input_ids_pad),
            "attention_mask": torch.tensor(attention_mask_pad),
            "labels": torch.tensor(labels_pad)
        }
