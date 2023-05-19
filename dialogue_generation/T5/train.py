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
from transformers import T5Tokenizer, T5ForConditionalGeneration

import params
from dataset import MyDataset
from utils import train_one_epoch, validate, read_json, init_logger

import warnings
warnings.filterwarnings("ignore")


def main(args):
    pprint(args.__dict__)

    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path)

    # data
    train_data = read_json(os.path.join(args.data_dir, "train.json"))
    valid_data = read_json(os.path.join(args.data_dir, "validation.json"))
    
    # num_workers
    num_workers = min([os.cpu_count(), args.train_batch_size if args.train_batch_size > 1 else 0, 8])

    # dataset, dataloader
    train_set = MyDataset(train_data, args.text_prefix, tokenizer)
    train_loader = DataLoader(train_set,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_workers,
                              collate_fn=train_set.collate_fn,
                              drop_last=True)

    valid_set = MyDataset(valid_data, args.text_prefix, tokenizer)
    valid_loader = DataLoader(valid_set,
                              batch_size=args.valid_batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=0,
                              collate_fn=valid_set.collate_fn,
                              drop_last=True)

    # model
    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_name_or_path)
    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=args.num_warmup_steps,
                                                   num_training_steps=len(train_loader) * args.train_batch_size)

    # tensorboard --logdir=runs
    # 用于记录训练过程中各个参数/指标的变化
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(sys.path[0], "runs", "{}_{}".format(args.pretrained_model_name_or_path, current_time))
    tb_writer = SummaryWriter(log_dir=log_dir)
    # logging
    logger = init_logger(args, current_time)

    os.makedirs(args.weights_dir, exist_ok=True)


    best_ppl = float('inf')
    for epoch in range(args.num_train_epochs):
        train_result = train_one_epoch(model=model,
                                       device=args.device,
                                       data_loader=train_loader,
                                       optimizer=optimizer,
                                       lr_scheduler=lr_scheduler,
                                       epoch=epoch)

        dev_result = validate(model=model,
                              device=args.device,
                              data_loader=valid_loader,
                              epoch=epoch)

        results = {
            'train_loss': train_result['loss'],
            'train_ppl': train_result['perplexity'],
            'dev_loss': dev_result['loss'],
            'dev_ppl': dev_result['perplexity'],
            'learning_rate': optimizer.param_groups[0]["lr"]
        }

        logger.info("=" * 100)
        # 记录训练中各个指标的信息
        for key, value in results.items():
            tb_writer.add_scalar(key, value, epoch)
            logger.info(f"{key}: {value}")

        # 保存在验证集上ppl最低的模型, 每次更新最低的ppl时就保存一个模型
        if dev_result['perplexity'] < best_ppl:
            torch.save(model.state_dict(), os.path.join(args.weights_dir, '{}-{}-epoch{}-ppl{:.3f}.pth'.format(
                args.pretrained_model_name_or_path, current_time, epoch, dev_result['perplexity'])))
            best_ppl = dev_result['perplexity']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model_name_or_path', type=str, default=params.pretrained_model_name_or_path)
    parser.add_argument('--text_prefix', type=str, default=params.text_prefix)
    parser.add_argument('--data_dir', type=str, default=params.data_dir)

    parser.add_argument('--num_train_epochs', type=int, default=params.num_train_epochs)
    parser.add_argument('--train_batch_size', type=int, default=params.train_batch_size)
    parser.add_argument('--valid_batch_size', type=int, default=params.valid_batch_size)
    parser.add_argument('--learning_rate', type=float, default=params.learning_rate)
    parser.add_argument('--weight_decay', type=float, default=params.weight_decay)
    parser.add_argument('--num_warmup_steps', type=int, default=params.num_warmup_steps)

    parser.add_argument('--device', default=params.device)
    parser.add_argument('--weights_dir', type=str, default=params.save_weights_path)

    args = parser.parse_args()

    main(args)

