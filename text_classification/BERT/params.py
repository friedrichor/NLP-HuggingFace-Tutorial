import os
import sys
import torch


num_classes = 4  # 类别数 (tweet_eval 中的情感分析数据集为 4 分类)

pretrained_model_name_or_path = 'bert-base-uncased'

num_train_epochs = 10
train_batch_size = 16
valid_batch_size = 8
learning_rate = 5e-4
weight_decay = 0
num_warmup_steps = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = os.path.join(sys.path[0], 'dataset')  # 数据集目录
weights_dir = os.path.join(sys.path[0], "weights")  # 模型权重保存目录
