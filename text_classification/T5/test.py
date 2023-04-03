import os
from pprint import pprint
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

import params
from dataset import MyDataset
from utils import read_json, test

import warnings
warnings.filterwarnings("ignore")


def main(args):
    pprint(args.__dict__)

    # data
    test_data_file = os.path.join(args.data_dir, "test.json")
    test_data = read_json(test_data_file)

    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path)

    # classes_map
    classes_map = read_json(args.classes_map_dir)

    # dataset, dataloader
    test_set = MyDataset(test_data, tokenizer, classes_map, args.prefix_text)
    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=0,
                             collate_fn=test_set.collate_fn,
                             drop_last=False)

    # model, load weights
    config = T5Config.from_pretrained(args.pretrained_model_name_or_path)
    weights_path = os.path.join(args.save_weights_path, args.weights_name)
    model = T5ForConditionalGeneration.from_pretrained(weights_path, config=config)
    model.to(args.device)

    labels_id_list = []
    for cls in classes_map.keys():
        labels_id = tokenizer.encode(text=cls, add_special_tokens=False)
        labels_id_list.append(labels_id[0])

    # test
    test_result = test(model=model, device=args.device, data_loader=test_loader, label_id_list=labels_id_list)
    print(test_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--classes_map_dir', type=str, default=params.classes_map_dir)
    parser.add_argument('--prefix_text', type=str, default=params.prefix_text)
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=params.pretrained_model_name_or_path)

    parser.add_argument('--device', default=params.device)
    parser.add_argument('--data_dir', type=str, default=params.data_dir)
    parser.add_argument('--save_weights_path', type=str, default=params.save_weights_path)
    parser.add_argument('--weights_name', type=str, default=None)

    args = parser.parse_args()

    main(args)

