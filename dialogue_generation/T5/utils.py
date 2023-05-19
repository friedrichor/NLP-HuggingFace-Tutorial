import os
import sys
import json
import logging
from tqdm import tqdm

import torch


def train_one_epoch(model, device, data_loader, optimizer, lr_scheduler, epoch):
    model.train()

    sum_loss = torch.zeros(1).to(device)  # cumulative loss
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)  # torch.Size([batch_size, max_seq_len(input_ids)])
        attention_mask = data['attention_mask'].to(device)  # torch.Size([batch_size, max_seq_len(input_ids)])
        labels = data['labels'].to(device)  # torch.Size([batch_size, max_seq_len(labels)])

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        sum_loss += loss.detach()
        avg_loss = sum_loss / (step + 1)  # tensor([???], device='cuda:0')
        perplexity = torch.exp(avg_loss)  # tensor([???], device='cuda:0')

        data_loader.desc = "[train epoch {}] lr: {:.5f}, loss: {:.3f}, ppl: {:.3f}".format(
            epoch,
            optimizer.param_groups[0]["lr"],
            avg_loss.item(),
            perplexity.item()
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return {
        'loss': avg_loss.item(),
        'perplexity': perplexity.item()
    }


@torch.no_grad()
def validate(model, device, data_loader, epoch):
    model.eval()

    sum_loss = torch.zeros(1).to(device)  # cumulative loss

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)  # torch.Size([batch_size, max_seq_len(input_ids)])
        attention_mask = data['attention_mask'].to(device)  # torch.Size([batch_size, max_seq_len(input_ids)])
        labels = data['labels'].to(device)  # torch.Size([batch_size, max_seq_len(labels)])

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # loss: float (not tensor)
        sum_loss += loss  # tensor([???], device='cuda:0')
        avg_loss = sum_loss / (step + 1)  # tensor([???], device='cuda:0')
        perplexity = torch.exp(avg_loss)  # tensor([???], device='cuda:0')

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, ppl: {:.3f}".format(
            epoch,
            avg_loss.item(),
            perplexity.item()
        )

    return {
        'loss': avg_loss.item(),
        'perplexity': perplexity.item()
    }


def read_json(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def init_logger(args, current_time):
    # 用于记录训练过程中的信息
    os.makedirs(os.path.join(sys.path[0], "logs"), exist_ok=True)
    log_path = os.path.join(sys.path[0], "logs", "{}_{}.txt".format(args.pretrained_model_name_or_path, current_time))
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    # write args information
    for key, value in args.__dict__.items():
        logger.info(f'{key}: {value}')

    return logger
