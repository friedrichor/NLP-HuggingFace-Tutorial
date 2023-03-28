import os
import sys
import torch


num_classes = 4  # 类别数 (tweet_eval 中的情感分析数据集为 4 分类)
classes_labels_dir = os.path.join(sys.path[0], 'classes_labels.json')

epochs = 5
batch_size = 16
lr = 5e-5
weight_decay = 5e-3
freeze_layers = True

pretrained_model_name_or_path = 'bert-base-uncased'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

data_dir = os.path.join(sys.path[0], 'dataset')  # 数据集目录
save_weights_path = os.path.join(sys.path[0], 'weights')  # 模型权重保存目录


# 以下模板参考于 PVLM
# Few-Shot Multi-Modal Sentiment Analysis with Prompt-Based Vision-Aware Language Modeling
# https://ieeexplore.ieee.org/document/9859654/
template1 = {
    'content': ['[CLS] the sentence " ', ' " has [MASK] emotion [SEP] '],
    'map': [0, 'x', 1]
}  # [CLS] the sentence " X " has [MASK] emotion [SEP]

template2 = {
    'content': [' [CLS] the sentence " ', ' " presents a [MASK] sentiment [SEP] '],
    'map': [0, 'x', 1]
}  # [CLS] the sentence " X " presents a [MASK] sentiment [SEP]
