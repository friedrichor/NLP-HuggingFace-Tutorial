import os
import sys
import argparse
import torch


parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--weight_decay', type=float, default=0)

TEXT_DIALOGUE_MODEL = ['microsoft/DialoGPT-small', 'microsoft/DialoGPT-medium', 'microsoft/DialoGPT-large']
parser.add_argument('--model_name', type=str, choices=TEXT_DIALOGUE_MODEL, default='microsoft/DialoGPT-small')
parser.add_argument('--tokenizer_name', type=str, default='microsoft/DialoGPT-small')

args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # num_workers
args.data_dir = os.path.join(sys.path[0], 'dataset_partial')  # 数据集目录
args.weights_dir = os.path.join(sys.path[0], 'weights')  # 模型权重保存目录
