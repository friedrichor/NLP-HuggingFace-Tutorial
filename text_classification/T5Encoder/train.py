import os
import sys
import logging
from pprint import pprint
from datetime import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import T5Tokenizer

import params
from dataset import MyDataset
from model import MyT5EncoderModel
from utils import read_json, train_one_epoch, validate

import warnings
warnings.filterwarnings("ignore")


def main(args):
    pprint(args.__dict__)
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    os.makedirs(args.save_weights_path, exist_ok=True)

    # tensorboard --logdir=runs
    # 用于记录训练过程中各个参数/指标的变化
    log_dir = os.path.join(sys.path[0], "runs", "{}_{}".format(args.pretrained_model_name_or_path.split('/')[-1], current_time))
    tb_writer = SummaryWriter(log_dir=log_dir)

    # logging
    # 用于记录训练过程中的信息
    os.makedirs(os.path.join(sys.path[0], "logs"), exist_ok=True)
    log_path = os.path.join(sys.path[0], "logs", "{}_{}.txt".format(args.pretrained_model_name_or_path.split('/')[-1], current_time))
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    for key, value in args.__dict__.items():
        logger.info(f'{key}: {value}')

    # data
    train_data_file = os.path.join(args.data_dir, "train.json")
    valid_data_file = os.path.join(args.data_dir, "validation.json")

    train_data = read_json(train_data_file)
    valid_data = read_json(valid_data_file)

    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path)  # AutoTokenizer

    # dataset, dataloader
    train_set = MyDataset(train_data, tokenizer)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=args.nw,
                              collate_fn=train_set.collate_fn,
                              drop_last=True)

    valid_set = MyDataset(valid_data, tokenizer)
    valid_loader = DataLoader(valid_set,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=0,
                              collate_fn=valid_set.collate_fn,
                              drop_last=False)

    # model
    model = MyT5EncoderModel(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                             num_classes=args.num_classes)
    model.to(args.device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(pg, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=len(train_loader),
                                                   num_training_steps=len(train_loader) * args.epochs)

    best_macro_f1 = 0
    for epoch in range(args.epochs):
        train_result = train_one_epoch(model=model,
                                       device=args.device,
                                       data_loader=train_loader,
                                       epoch=epoch,
                                       optimizer=optimizer,
                                       lr_scheduler=lr_scheduler)

        dev_result = validate(model=model,
                              device=args.device,
                              data_loader=valid_loader,
                              epoch=epoch)

        results = {
            'learning_rate': optimizer.param_groups[0]["lr"],
            'train_loss': train_result['loss'],
            'train_accuracy': train_result['accuracy'],
            'train_macro_f1': train_result['macro_f1'],
            'train_micro_f1': train_result['micro_f1'],
            'train_weighted_f1': train_result['weighted_f1'],
            'dev_loss': dev_result['loss'],
            'dev_accuracy': dev_result['accuracy'],
            'dev_macro_f1': dev_result['macro_f1'],
            'dev_micro_f1': dev_result['micro_f1'],
            'dev_weighted_f1': dev_result['weighted_f1']
        }

        logger.info("=" * 100)
        logger.info(f"epoch: {epoch}")
        # 记录训练中各个指标的信息
        for key, value in results.items():
            tb_writer.add_scalar(key, value, epoch)
            logger.info(f"{key}: {value}")

        # 保存在验证集上 macro_f1 最高的模型
        if dev_result['macro_f1'] > best_macro_f1:
            torch.save(model.state_dict(), os.path.join(args.save_weights_path, '{}-{}-epoch{}.pth'.format(
                       args.pretrained_model_name_or_path.split('/')[-1], current_time, epoch)))
            best_macro_f1 = dev_result['macro_f1']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=params.num_classes)
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=params.pretrained_model_name_or_path)

    parser.add_argument('--epochs', type=int, default=params.epochs)
    parser.add_argument('--batch_size', type=int, default=params.batch_size)
    parser.add_argument('--lr', type=float, default=params.lr)
    parser.add_argument('--weight_decay', type=float, default=params.weight_decay)

    parser.add_argument('--device', default=params.device)
    parser.add_argument('--nw', type=int, default=params.num_workers)
    parser.add_argument('--data_dir', type=str, default=params.data_dir)
    parser.add_argument('--save_weights_path', type=str, default=params.save_weights_path)

    args = parser.parse_args()

    main(args)

