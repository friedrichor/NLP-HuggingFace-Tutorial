import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        text = self.data[index]["text"]
        label = self.data[index]["label"]

        text_encoded = self.tokenizer.encode_plus(text=text,
                                                  add_special_tokens=True,  # T5应该是没有[CLS]和[SEP]的，有个结束符</s>，如果需要其他标识符请自行添加
                                                  return_attention_mask=True)

        return {
            "input_ids": text_encoded["input_ids"],
            "attention_mask": text_encoded["attention_mask"],
            "label": label
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids_list = [torch.tensor(instance['input_ids']) for instance in batch]
        input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        attention_mask_list = [torch.tensor(instance['attention_mask']) for instance in batch]
        attention_mask_pad = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        label_list = [torch.tensor(instance['label']) for instance in batch]

        return {
            "input_ids": torch.tensor(input_ids_pad),
            "attention_mask": torch.tensor(attention_mask_pad),
            "label": torch.tensor(label_list)
        }
