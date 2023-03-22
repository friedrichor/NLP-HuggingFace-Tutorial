"""
Load the entire dataset and format it and save it to disk
"""

import os
import sys
import json
from tqdm import tqdm
from pprint import pprint

from datasets import load_dataset


def process_data(data, mode: str):
    """Process and save data to disk
    :param data: train/validation/test set loaded from huggingface
    :param mode: Choose from [“train”, "validation", "test"]. For file naming
    :return: None
    """
    print(f"There are {len(data)} pieces of data in the {mode} set")

    dialogue_dic_list = []
    for per_data in tqdm(data):
        context = per_data['context']
        response = per_data['response']
        dialogue = context + "<|endoftext|> " + response + "<|endoftext|>"
        dialogue_dic_list.append({"dialogue": dialogue})

    # Writing to a json file
    dataset_folder = os.path.join(sys.path[0], 'dataset')
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    fout_path = os.path.join(dataset_folder, mode + '.json')
    with open(fout_path, 'w', encoding='utf-8') as fout:
        json.dump(dialogue_dic_list, fout, indent=4)


def process_test_data_for_evaluate(data):
    dialogue_dic_list = []
    for per_data in tqdm(data):
        context = per_data['context']
        response = per_data['response']
        dialogue_dic_list.append({"context": context, "response": response})

    # Writing to a json file
    dataset_folder = os.path.join(sys.path[0], 'dataset')
    os.makedirs(dataset_folder, exist_ok=True)
    fout_path = os.path.join(dataset_folder, 'test_evaluate.json')
    with open(fout_path, 'w', encoding='utf-8') as fout:
        json.dump(dialogue_dic_list, fout, indent=4)


if __name__ == '__main__':
    # 使用 ProsocialDialog 数据集
    # https://huggingface.co/datasets/allenai/prosocial-dialog
    dataset = load_dataset("allenai/prosocial-dialog")
    print(dataset)  # 查看数据集格式
    pprint(dataset['train'][0], width=1000)  # 查看数据格式
    print("=" * 100)

    train_data = dataset['train']
    process_data(train_data, 'train')

    validation_data = dataset['validation']
    process_data(validation_data, 'validation')

    test_data = dataset['test']
    process_data(test_data, 'test')
    process_test_data_for_evaluate(test_data)


