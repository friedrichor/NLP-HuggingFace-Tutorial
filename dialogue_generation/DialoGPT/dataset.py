import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# TDRG -- Textual Dialogue Response Generator
class TDRGDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        dialogue = self.data[index]["dialogue"]

        dialogue_encoded = self.tokenizer.encode(text=dialogue,
                                                 add_special_tokens=False,
                                                 padding='max_length',
                                                 truncation=True,
                                                 max_length=256,
                                                 return_tensors='pt')

        return {
            'input_ids': dialogue_encoded
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids_list = [instance['input_ids'] for instance in batch]
        input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        return {
            'input_ids': torch.tensor(input_ids_pad)
        }
