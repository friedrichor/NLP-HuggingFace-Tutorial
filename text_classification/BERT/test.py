import os
from pprint import pprint
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

import params
from dataset import MyDataset
from model import MyBertModel_CLS, MyBertModel_Pool
from utils import read_json, validate

import warnings
warnings.filterwarnings("ignore")


def main(args):
    pprint(args.__dict__)

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    
    # data
    test_data = read_json(os.path.join(args.data_dir, "test.json"))

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
    model = MyBertModel_CLS(args.pretrained_model_name_or_path, args.num_classes)
    """
    根据实际使用的模型更改
    model = MyBertModel_Pool(args.pretrained_model_name_or_path, args.num_classes)
    model = BertForSequenceClassification.from_pretrained(args.pretrained_model_name_or_path, args.num_classes)
    """
    # load weights
    weights_path = os.path.join(args.weights_dir, args.weights_name)
    model.load_state_dict(torch.load(weights_path, map_location=args.device))
    model.to(args.device)

    # test
    test_result = validate(model=model, device=args.device, data_loader=test_loader)
    print(test_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=params.num_classes)
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=params.pretrained_model_name_or_path)
    parser.add_argument('--data_dir', type=str, default=params.data_dir)

    parser.add_argument('--device', default=params.device)
    parser.add_argument('--weights_dir', type=str, default=params.weights_dir)
    parser.add_argument('--weights_name', type=str, default=None)

    args = parser.parse_args()

    main(args)

