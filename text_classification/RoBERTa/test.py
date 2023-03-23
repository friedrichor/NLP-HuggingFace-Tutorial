import os
from pprint import pprint
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import params
from dataset import MyDataset
from model import MyRobertaModel
from utils import read_json, validate

import warnings
warnings.filterwarnings("ignore")


def main(args):
    pprint(args.__dict__)

    # data
    test_data_file = os.path.join(args.data_dir, "test.json")
    test_data = read_json(test_data_file)

    # tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model_name_or_path)  # AutoTokenizer

    # dataset, dataloader
    test_set = MyDataset(test_data, tokenizer)
    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=0,
                             collate_fn=test_set.collate_fn,
                             drop_last=False)

    # model
    model = MyRobertaModel(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                           num_classes=args.num_classes)
    """
    # 此外也可以直接调用 RobertaForSequenceClassification，并更改其输出维度为 num_classes
    # 直接使用以下代码代替上方代码即可
    model = RobertaForSequenceClassification.from_pretrained(args.pretrained_model_name_or_path, num_labels=args.num_classes)
    """
    # load weights
    weights_path = os.path.join(args.save_weights_path, args.weights_name)
    model.load_state_dict(torch.load(weights_path, map_location=args.device))
    model.to(args.device)

    # test
    test_result = validate(model=model, device=args.device, data_loader=test_loader)
    print(test_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=params.num_classes)
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=params.pretrained_model_name_or_path)

    parser.add_argument('--device', default=params.device)
    parser.add_argument('--data_dir', type=str, default=params.data_dir)
    parser.add_argument('--save_weights_path', type=str, default=params.save_weights_path)
    parser.add_argument('--weights_name', type=str, default=None)

    args = parser.parse_args()

    main(args)

