import os
import sys
import logging
from pprint import pprint
from datetime import datetime
import argparse

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import Adafactor, AdafactorSchedule, AdamW, get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration

import params
from dataset import MyDataset
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
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path)

    # classes_map
    classes_map = read_json(args.classes_map_dir)
    # labels_id_list
    label_id_list = []
    for cls in classes_map.keys():
        labels_id = tokenizer.encode(text=cls, add_special_tokens=False)
        label_id_list.append(labels_id[0])

    # dataset, dataloader
    train_set = MyDataset(train_data, tokenizer, classes_map, args.prefix_text)

    if args.use_weighted_random_sampler:
        data_label_list = []
        for data in train_set:
            data_label_list.append(data['labels'][0])
        weights = [1.0 / data_label_list.count(label) for label in data_label_list]
        sampler = WeightedRandomSampler(weights, len(train_data), replacement=False)
        print("Completed sampling.")

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=False if args.use_weighted_random_sampler else True,
                              sampler=sampler if args.use_weighted_random_sampler else None,
                              pin_memory=True,
                              num_workers=args.num_workers,
                              collate_fn=train_set.collate_fn,
                              drop_last=True)

    valid_set = MyDataset(valid_data, tokenizer, classes_map, args.prefix_text)
    valid_loader = DataLoader(valid_set,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=0,
                              collate_fn=valid_set.collate_fn,
                              drop_last=False)

    # model
    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_name_or_path)
    model.to(args.device)

    if args.use_Adafactor and args.use_AdafactorSchedule:
        # https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/optimizer_schedules#transformers.Adafactor
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        lr_scheduler = AdafactorSchedule(optimizer)
    elif args.use_Adafactor and not args.use_AdafactorSchedule:
        optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=args.learning_rate)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=args.lr_warmup_steps,
                                                       num_training_steps=len(train_loader) * args.num_train_epochs)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=args.lr_warmup_steps,
                                                       num_training_steps=len(train_loader) * args.num_train_epochs)

    best_macro_f1 = 0
    for epoch in range(args.num_train_epochs):
        train_result = train_one_epoch(model=model,
                                       device=args.device,
                                       data_loader=train_loader,
                                       epoch=epoch,
                                       optimizer=optimizer,
                                       lr_scheduler=lr_scheduler)

        dev_result = validate(model=model,
                              device=args.device,
                              data_loader=valid_loader,
                              label_id_list=label_id_list,
                              epoch=epoch)

        results = {
            'learning_rate': optimizer.param_groups[0]["lr"],
            'train_loss': train_result['loss'],
            'train_accuracy': train_result['accuracy'],
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

    parser.add_argument('--classes_map_dir', type=str, default=params.classes_map_dir)
    parser.add_argument('--prefix_text', type=str, default=params.prefix_text)
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=params.pretrained_model_name_or_path)

    parser.add_argument('--num_train_epochs', type=int, default=params.num_train_epochs)
    parser.add_argument('--batch_size', type=int, default=params.batch_size)
    parser.add_argument(
        "--use_weighted_random_sampler", default=False, help="Whether or not to use WeightedRandomSampler."
    )

    parser.add_argument('--device', default=params.device)
    parser.add_argument('--num_workers', type=int, default=params.num_workers)
    parser.add_argument('--data_dir', type=str, default=params.data_dir)
    parser.add_argument('--save_weights_path', type=str, default=params.save_weights_path)

    parser.add_argument(
        "--use_Adafactor", default=True, help="Whether or not to use Adafactor."
    )
    parser.add_argument(
        "--use_AdafactorSchedule", default=True, help="Whether or not to use AdafactorSchedule."
    )
    parser.add_argument(
        '--learning_rate', type=float, default=params.learning_rate,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=params.lr_warmup_steps,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        '--weight_decay', type=float, default=params.weight_decay, help="Weight decay to use."
    )

    args = parser.parse_args()

    main(args)

