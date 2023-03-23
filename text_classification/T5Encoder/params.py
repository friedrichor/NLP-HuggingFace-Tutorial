import os
import sys
import torch


num_classes = 4  # 类别数 (tweet_eval 中的情感分析数据集为 4 分类)

epochs = 10
batch_size = 16
lr = 5e-4
weight_decay = 5e-3

pretrained_model_name_or_path = 't5-base'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

data_dir = os.path.join(sys.path[0], 'dataset')  # 数据集目录
save_weights_path = os.path.join(sys.path[0], 'weights')  # 模型权重保存目录

