import os
import sys
import torch


pretrained_model_name_or_path = "microsoft/DialoGPT-small"

num_train_epochs = 10
train_batch_size = 8
valid_batch_size = 8
learning_rate = 5e-4
weight_decay = 0
num_warmup_steps = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = os.path.join(sys.path[0], "partial_dataset")  # 数据集目录 (部分数据)
# data_dir = os.path.join(sys.path[0], "dataset")  # 数据集目录 (全部数据)
weights_dir = os.path.join(sys.path[0], "weights")  # 模型权重保存目录

