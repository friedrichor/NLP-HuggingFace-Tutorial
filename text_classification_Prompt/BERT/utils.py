import sys
import json
from tqdm import tqdm

import torch
from sklearn.metrics import accuracy_score, f1_score


def read_json(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def train_one_epoch(model, device, data_loader, epoch, optimizer, lr_scheduler):
    model.train()

    predicted_labels = torch.LongTensor([]).to(device)
    ground_truth_labels = torch.LongTensor([]).to(device)

    sum_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        loss, pred_labels, gt_labels = model(input_ids, attention_mask, labels)

        ground_truth_labels = torch.cat([ground_truth_labels, gt_labels])
        predicted_labels = torch.cat([predicted_labels, pred_labels.to(device)])

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
        accuracy = accuracy_score(ground_truth_labels.tolist(), predicted_labels.tolist())
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        macro_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='macro')
        micro_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='micro')
        weighted_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='weighted')

        loss.backward()

        sum_loss += loss.detach()
        avg_loss = sum_loss.item() / (step + 1)

        data_loader.desc = "[train epoch {}] lr: {:.5f}, loss: {:.3f}, acc: {:.3f}, macro_f1: {:.3f}, micro_f1: {:.3f}, weighted_f1: {:.3f}".format(
            epoch, optimizer.param_groups[0]["lr"], avg_loss,
            accuracy, macro_f1, micro_f1, weighted_f1
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

    sum_loss = torch.zeros(1).to(device)  # 累计损失

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        loss, pred_labels, gt_labels = model(input_ids, attention_mask, labels)

        ground_truth_labels = torch.cat([ground_truth_labels, gt_labels])
        predicted_labels = torch.cat([predicted_labels, pred_labels.to(device)])

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
        accuracy = accuracy_score(ground_truth_labels.tolist(), predicted_labels.tolist())
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        macro_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='macro')
        micro_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='micro')
        weighted_f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), average='weighted')

        sum_loss += loss.detach()
        avg_loss = sum_loss.item() / (step + 1)

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}, macro_f1: {:.3f}, micro_f1: {:.3f}, weighted_f1: {:.3f}".format(
            epoch, avg_loss,
            accuracy, macro_f1, micro_f1, weighted_f1
        )

    return {
        'loss': avg_loss,
        'accuracy':accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1
    }