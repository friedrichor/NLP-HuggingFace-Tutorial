from typing import Dict

import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput


class MyBertMLMModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str, tokenizer: BertTokenizer, classes_labels: Dict):
        super().__init__()
        self.lm_model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        self.labels_id_list = tokenizer.convert_tokens_to_ids(list(classes_labels.keys()))

    def forward(self, input_ids, attention_mask, labels):
        output = self.lm_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = output.loss, output.logits  # logits 得到序列中每一个词的概率
                                                   # logits: torch.Size([batch size, sequence length, vocab size])
        pred = logits[labels != -100]  # 把labels中为-100的(非MASK位置)都去掉, size=[batch size, vocab size]
        probs = pred[:, self.labels_id_list]  # 只得到"anger"、"joy"、"optimism"、"sadness"四个单词的概率, size=[batch size, 4]
        pred_labels_idx = torch.argmax(probs, dim=-1).tolist()  # 最后一个维度的最大值的索引

        pred_labels = [self.labels_id_list[i] for i in pred_labels_idx]  # predicted labels, size=[batch size]
        gt_labels = labels[labels != -100]  # ground-truth labels, size=[batch size]

        return loss, torch.tensor(pred_labels), gt_labels






