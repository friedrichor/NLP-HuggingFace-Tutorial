import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MyDataset(Dataset):
    def __init__(self, data, tokenizer, template, classes_labels):
        self.data = data
        self.tokenizer = tokenizer

        self.template = template
        self.labels_map = {value: key for key, value in classes_labels.items()}  # {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}

    def apply_template(self, text):
        new_text = ""
        for i in self.template['map']:
            if isinstance(i, int):
                new_text += self.template['content'][i]
            elif i == 'x':
                new_text += text

        return new_text

    def __getitem__(self, index):
        text = self.data[index]["text"]
        label = self.data[index]["label"]

        text = self.apply_template(text)  # 套用模板
        text_encoded = self.tokenizer.encode_plus(text=text,
                                                  add_special_tokens=False,
                                                  return_attention_mask=True)

        input_ids, attention_mask = text_encoded["input_ids"], text_encoded["attention_mask"]
        labels = [-100] * len(input_ids)

        for i, id in enumerate(input_ids):
            if id == self.tokenizer.mask_token_id:
                labels[i] = self.tokenizer._convert_token_to_id(self.labels_map[label])

        return {
            "input_ids": text_encoded["input_ids"],
            "attention_mask": text_encoded["attention_mask"],
            "labels": labels
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids_list = [torch.tensor(instance['input_ids']) for instance in batch]
        input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        attention_mask_list = [torch.tensor(instance['attention_mask']) for instance in batch]
        attention_mask_pad = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        labels_list = [torch.tensor(instance['labels']) for instance in batch]
        labels_pad = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        return {
            "input_ids": torch.tensor(input_ids_pad),
            "attention_mask": torch.tensor(attention_mask_pad),
            "labels": torch.tensor(labels_pad)
        }

