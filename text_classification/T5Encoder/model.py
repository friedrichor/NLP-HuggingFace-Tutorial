import torch
import torch.nn as nn
from transformers import T5EncoderModel
from transformers.modeling_outputs import SequenceClassifierOutput


class MyT5EncoderModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path, num_classes):
        super().__init__()
        self.backbone = T5EncoderModel.from_pretrained(pretrained_model_name_or_path)

        out_features = self.backbone.encoder.block[-1].layer[-1].DenseReluDense.wo.out_features  # t5-base 为 768，t5-large 为 1024
        self.fc1 = nn.Linear(out_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask):
        out_backbone = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        out_backbone = out_backbone.last_hidden_state  # [batch_size, seq_len, out_features]
        out_backbone = torch.mean(out_backbone, dim=1)  # [batch_size, out_features]
        out_fc1 = self.fc1(out_backbone)  # [batch_size, 128]
        out_fc2 = self.fc2(out_fc1)  # [batch_size, num_classes]

        return SequenceClassifierOutput(logits=out_fc2)
