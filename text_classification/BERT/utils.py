import os
import sys
import json
import logging
from tqdm import tqdm

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score




def train_one_epoch(model, device, data_loader, epoch, optimizer, lr_scheduler):
    model.train()

    predicted_labels = torch.LongTensor([]).to(device)
    ground_truth_labels = torch.LongTensor([]).to(device)

    loss_function = nn.CrossEntropyLoss()
    sum_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        label = data['label'].to(device)

        pred_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_label = torch.max(pred_logits, dim=1)[1]

        ground_truth_labels = torch.cat([ground_truth_labels, label])
        predicted_labels = torch.cat([predicted_labels, pred_label])

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
        accuracy = accuracy_score(ground_truth_labels.tolist(), predicted_labels.tolist())
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        macro_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='macro')
        micro_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='micro')
        weighted_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='weighted')

        loss = loss_function(pred_logits, label.to(device))
        loss.backward()

        sum_loss += loss.detach()
        avg_loss = sum_loss.item() / (step + 1)

        data_loader.desc = "[train epoch {}] lr: {:.5f}, loss: {:.3f}, acc: {:.3f}, macro_f1: {:.3f}, micro_f1: {:.3f}".format(
            epoch, optimizer.param_groups[0]["lr"], avg_loss,
            accuracy, macro_f1, micro_f1
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
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1
    }


@torch.no_grad()
def validate(model, device, data_loader, epoch=0):
    model.eval()

    predicted_labels = torch.LongTensor([]).to(device)
    ground_truth_labels = torch.LongTensor([]).to(device)

    loss_function = nn.CrossEntropyLoss()
    sum_loss = torch.zeros(1).to(device)  # 累计损失

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        label = data['label'].to(device)

        pred_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_label = torch.max(pred_logits, dim=1)[1]

        ground_truth_labels = torch.cat([ground_truth_labels, label])
        predicted_labels = torch.cat([predicted_labels, pred_label])

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
        accuracy = accuracy_score(ground_truth_labels.tolist(), predicted_labels.tolist())
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        macro_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='macro')
        micro_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='micro')
        weighted_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='weighted')

        loss = loss_function(pred_logits, label.to(device))
        sum_loss += loss.detach()
        avg_loss = sum_loss.item() / (step + 1)

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}, macro_f1: {:.3f}, micro_f1: {:.3f}".format(
            epoch, avg_loss,
            accuracy, macro_f1, micro_f1
        )

    return {
        'loss': avg_loss,
        'accuracy':accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1
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