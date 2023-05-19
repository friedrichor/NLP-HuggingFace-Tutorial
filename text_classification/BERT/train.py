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
from transformers import BertTokenizer, BertForSequenceClassification

import params
from dataset import MyDataset
from model import MyBertModel_CLS, MyBertModel_Pool
from utils import train_one_epoch, validate, read_json, init_logger

import warnings
warnings.filterwarnings("ignore")


def main(args):
    pprint(args.__dict__)
    
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    
    # data
    train_data = read_json(os.path.join(args.data_dir, "train.json"))
    valid_data = read_json(os.path.join(args.data_dir, "validation.json"))
    
    # num_workers
    num_workers = min([os.cpu_count(), args.train_batch_size if args.train_batch_size > 1 else 0, 8])

    # dataset, dataloader
    train_set = MyDataset(train_data, tokenizer)
    train_loader = DataLoader(train_set,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_workers,
                              collate_fn=train_set.collate_fn,
                              drop_last=True)

    valid_set = MyDataset(valid_data, tokenizer)
    valid_loader = DataLoader(valid_set,
                              batch_size=args.valid_batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=0,
                              collate_fn=valid_set.collate_fn,
                              drop_last=True)

    # model
    # 这里调用的是我更改的模型 (仅作为一个示例)，取出 [CLS] token 的表征输入给两个全连接层，且保证最终输出维度为 num_classes
    # 使用 [CLS] 用于分类是 BERT 用来做分类的默认方法
    model = MyBertModel_CLS(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                            num_classes=args.num_classes)
    model.to(args.device)
    """
    # 这里是另一种更改的模型，将整段文本的表征输入给两个全连接层，且保证最终输出维度为 num_classes
    model = MyBertModel_Pool(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                             num_classes=args.num_classes)
    model.to(args.device)
    """
    """
    # 此外也可以直接调用 BertForSequenceClassification,并更改其最后一层输出维度为 num_classes
    # 直接使用以下代码代替上方代码即可
    model = BertForSequenceClassification.from_pretrained(args.pretrained_model_name_or_path, num_labels=args.num_classes)
    model.to(args.device)
    """
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=args.num_warmup_steps,
                                                   num_training_steps=len(train_loader) * args.num_train_epochs)

    # tensorboard --logdir=runs
    # 用于记录训练过程中各个参数/指标的变化
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(sys.path[0], "runs", "{}_{}".format(args.pretrained_model_name_or_path, current_time))
    tb_writer = SummaryWriter(log_dir=log_dir)
    # logging
    logger = init_logger(args, current_time)
    
    os.makedirs(args.weights_dir, exist_ok=True)

    best_macro_f1 = 0.0
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
        # 记录训练中各个指标的信息
        for key, value in results.items():
            tb_writer.add_scalar(key, value, epoch)
            logger.info(f"{key}: {value}")

        # 保存在验证集上 macro_f1 最高的模型
        if dev_result['macro_f1'] > best_macro_f1:
            torch.save(model.state_dict(), os.path.join(args.weights_dir, '{}-{}-epoch{}-macro_f1{:.3f}.pth'.format(
                       args.pretrained_model_name_or_path, current_time, epoch, dev_result['macro_f1'])))
            best_macro_f1 = dev_result['macro_f1']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=params.num_classes)
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=params.pretrained_model_name_or_path)
    parser.add_argument('--data_dir', type=str, default=params.data_dir)

    parser.add_argument('--num_train_epochs', type=int, default=params.num_train_epochs)
    parser.add_argument('--train_batch_size', type=int, default=params.train_batch_size)
    parser.add_argument('--valid_batch_size', type=int, default=params.valid_batch_size)
    parser.add_argument('--learning_rate', type=float, default=params.learning_rate)
    parser.add_argument('--weight_decay', type=float, default=params.weight_decay)
    parser.add_argument('--num_warmup_steps', type=int, default=params.num_warmup_steps)

    parser.add_argument('--device', default=params.device)
    parser.add_argument('--weights_dir', type=str, default=params.weights_dir)

    args = parser.parse_args()

    main(args)

