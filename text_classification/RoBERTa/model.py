import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput


class MyRobertaModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path, num_classes, freeze_layers=False):
        super().__init__()
        self.backbone = RobertaModel.from_pretrained(pretrained_model_name_or_path)

        if freeze_layers:
            for name, para in self.backbone.named_parameters():
                if 'pooler' not in name:  # 除了最后一个 pooler 层外，其它层全部冻结
                    para.requires_grad_(False)

        self.fc1 = nn.Linear(self.backbone.pooler.dense.out_features, 128)  # [768, 128]
        self.fc2 = nn.Linear(128, num_classes)  # [768, num_classes]

    def forward(self, input_ids, attention_mask):
        out_backbone = self.backbone(input_ids=input_ids, attention_mask=attention_mask).pooler_output  # [batch_size, 768]
        out_fc1 = self.fc1(out_backbone)  # [batch_size, 128]
        out_fc2 = self.fc2(out_fc1)  # [batch_size, num_classes]

        return SequenceClassifierOutput(logits=out_fc2)
