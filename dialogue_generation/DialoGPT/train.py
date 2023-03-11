import os
import sys
import logging
from pprint import pprint
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from params import args
from dataset import TDRGDataset
from utils import tokenizer_plus, read_json, train_one_epoch, validate

import warnings
warnings.filterwarnings("ignore")


def main():
    pprint(args.__dict__)

    if not os.path.exists(args.weights_dir):
        os.makedirs(args.weights_dir)

    # tensorboard --logdir=runs
    # 用于记录训练过程中各个参数/指标的变化
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(sys.path[0], "runs",
                           "{}_{}_lr{}_wd{}".format(args.model_name.split('/')[-1], current_time, str(args.lr),
                                                    str(args.weight_decay)))
    tb_writer = SummaryWriter(log_dir=log_dir)

    # logging
    # 用于记录训练过程中的信息
    if not os.path.exists(os.path.join(sys.path[0], "logs")):
        os.makedirs(os.path.join(sys.path[0], "logs"))
    log_path = os.path.join(sys.path[0], "logs", "{}_{}.txt".format(args.model_name.split('/')[-1], current_time))
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
    # 添加 pad_token 和 sep_token (DialoGPT 的 Tokenizer 中没有 pad_token 和 sep_token)
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name)  # AutoTokenizer
    tokenizer, _ = tokenizer_plus(tokenizer, logger)

    # dataset, dataloader
    train_set = TDRGDataset(train_data, tokenizer)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=args.nw,
                              collate_fn=train_set.collate_fn,
                              drop_last=True)

    valid_set = TDRGDataset(valid_data, tokenizer)
    valid_loader = DataLoader(valid_set,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=args.nw,
                              collate_fn=valid_set.collate_fn,
                              drop_last=False)

    # model
    model = GPT2LMHeadModel.from_pretrained(args.model_name).to(args.device)  # AutoModelForCausalLM
    model.resize_token_embeddings(len(tokenizer))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(pg, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=len(train_loader),
                                                   num_training_steps=len(train_loader) * args.epochs)

    best_ppl = float('inf')
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
            current_time = datetime.now().strftime("%b%d_%H-%M-%S")
            torch.save(model.state_dict(), os.path.join(args.weights_dir, '{}-{}-epoch{}-ppl{:.3f}.pth'.format(
                       args.model_name.split('/')[-1], current_time, epoch, dev_result['perplexity'])))
            best_ppl = dev_result['perplexity']


if __name__ == "__main__":
    main()
