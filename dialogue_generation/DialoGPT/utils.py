import sys
import json
from tqdm import tqdm

import torch


def tokenizer_plus(tokenizer, logger=None):
    """
    由于 DialoGPT 的 Tokenizer 中没有 pad_token 和 sep_token, 在这里添加上
    """
    num_add_token = 0
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        num_add_token += 1
        if logger:
            logger.info("add_special_tokens: [PAD]")

    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        num_add_token += 1
        if logger:
            logger.info("add_special_tokens: [SEP]")

    return tokenizer, num_add_token


def read_json(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def train_one_epoch(model, device, data_loader, epoch, optimizer, lr_scheduler):
    model.train()

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        inputs = data['input_ids'].to(device)  # [batch size, max seq len(batch内句子最大长度)]
        # labels = data['label'].to(device)

        outputs = model(input_ids=inputs, labels=inputs)  # 注意这里指定 input_ids 和 labels 相同
        # outputs: [0]: loss        一个数
        #          [1]: logits      size=[batch size, max seq len, vocab size], 如[16,512,50257]
        loss = outputs[0]

        loss.backward()
        accu_loss += loss.detach()
        avg_loss = accu_loss.item() / (step + 1)
        perplexity = torch.exp(torch.tensor(avg_loss))

        data_loader.desc = "[train epoch {}] lr: {:.5f}, loss: {:.3f}, ppl: {:.3f}".format(
            epoch,
            optimizer.param_groups[0]["lr"],
            avg_loss,
            perplexity
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return {
        'loss': avg_loss,
        'perplexity': perplexity
    }


@torch.no_grad()
def validate(model, device, data_loader, epoch):
    model.eval()

    accu_loss = torch.zeros(1).to(device)  # 累计损失

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        inputs = data['input_ids'].to(device)  # [batch size, max seq len(batch内句子最大长度)]

        outputs = model(input_ids=inputs, labels=inputs)
        # outputs: [0]: loss        一个数
        #          [1]: logits      size=[batch size, max seq len, vocab size], 如[16,512,50257]
        loss = outputs[0]

        accu_loss += loss.mean().item()
        avg_loss = accu_loss / (step + 1)
        perplexity = torch.exp(torch.tensor(avg_loss))

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, ppl: {:.3f}".format(
            epoch,
            avg_loss.item(),
            perplexity.item()
        )

    return {
        'loss': avg_loss.item(),
        'perplexity': perplexity.item()
    }