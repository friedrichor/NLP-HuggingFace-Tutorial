import os
import sys
import torch

epochs = 10
batch_size = 8
lr = 5e-4
weight_decay = 0

model_name = 'microsoft/DialoGPT-small'
tokenizer_name = 'microsoft/DialoGPT-small'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

data_dir = os.path.join(sys.path[0], 'dataset_partial')  # 数据集目录
weights_dir = os.path.join(sys.path[0], 'weights')  # 模型权重保存目录

