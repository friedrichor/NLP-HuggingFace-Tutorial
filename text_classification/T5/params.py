import os
import sys
import torch


classes_map_dir = os.path.join(sys.path[0], "classes_map.json")
prefix_text = "tweet_eval emotion sentence: "

pretrained_model_name_or_path = 't5-base'

num_train_epochs = 10
batch_size = 16
learning_rate = 1e-4
lr_warmup_steps = 0
weight_decay = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

data_dir = os.path.join(sys.path[0], 'dataset')  # 数据集目录
save_weights_path = os.path.join(sys.path[0], 'weights')  # 模型权重保存目录

