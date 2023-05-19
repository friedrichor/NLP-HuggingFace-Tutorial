import torch
import torch.nn as nn
from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class MyBertModel_CLS(nn.Module):
    def __init__(self, pretrained_model_name_or_path, num_classes):
        super().__init__()
        self.backbone = BertModel.from_pretrained(pretrained_model_name_or_path)

        self.fc1 = nn.Linear(768, 128)  # 768 -> 128
        self.fc2 = nn.Linear(128, num_classes)  # 128 -> 2

    def forward(self, input_ids, attention_mask):
        out_backbone = self.backbone(input_ids=input_ids, 
                                     attention_mask=attention_mask).last_hidden_state  # [batch_size, sequence_length, 768]
        cls_hidden_state = out_backbone[:, 0, :]  # 取出 [CLS] token 的表征
        out_fc1 = self.fc1(cls_hidden_state)  # [batch_size, 128]
        out_fc2 = self.fc2(out_fc1)  # [batch_size, num_classes]

        return SequenceClassifierOutput(logits=out_fc2)


class MyBertModel_Pool(nn.Module):
    def __init__(self, pretrained_model_name_or_path, num_classes):
        super().__init__()
        self.backbone = BertModel.from_pretrained(pretrained_model_name_or_path)

        self.fc1 = nn.Linear(self.backbone.pooler.dense.out_features, 128)  # [768, 128]
        self.fc2 = nn.Linear(128, num_classes)  # [768, num_classes]

    def forward(self, input_ids, attention_mask):
        out_backbone = self.backbone(input_ids=input_ids, attention_mask=attention_mask).pooler_output  # [batch_size, 768]
        out_fc1 = self.fc1(out_backbone)  # [batch_size, 128]
        out_fc2 = self.fc2(out_fc1)  # [batch_size, num_classes]

        return SequenceClassifierOutput(logits=out_fc2)
