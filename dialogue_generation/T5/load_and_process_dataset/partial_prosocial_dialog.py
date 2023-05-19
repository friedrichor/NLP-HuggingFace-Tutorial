"""
Load the {partial} dataset and format it and save it to disk
"""
import os
import sys
import json
from tqdm import tqdm
from pprint import pprint

from datasets import load_dataset


NUM_DATA_USED = {"train": 4000, "validation": 4000, "test": 500}


def process_data(data, mode: str, num: int):
    """Process and save data to disk
    :param data: train/validation/test set loaded from huggingface
    :param mode: Choose from [“train”, "validation", "test"]. For file naming
    :return: None
    """
    print(f"There are {len(data)} pieces of data in the {mode} set")

    dialogue_dic_list = []
    for i, per_data in tqdm(enumerate(data)):
        if i == num:
            break
        context = per_data['context']
        response = per_data['response']
        dialogue_dic_list.append({"context": context, "response": response})

    dataset_folder = os.path.join(sys.path[0], '../partial_dataset')
    os.makedirs(dataset_folder, exist_ok=True)
    # Writing to a json file
    with open(os.path.join(dataset_folder, mode + '.json'), 'w', encoding='utf-8') as f:
        json.dump(dialogue_dic_list, f, indent=4)


if __name__ == '__main__':
    # 使用 ProsocialDialog 数据集(单轮对话数据集)
    # https://huggingface.co/datasets/allenai/prosocial-dialog
    dataset = load_dataset("allenai/prosocial-dialog")

    print(dataset)  # 查看数据集格式
    pprint(dataset['train'][0], width=1000)  # 查看数据格式
    print("=" * 100)

    for mode in ["train", "validation", "test"]:
        splited_data = dataset[mode]
        process_data(splited_data, mode, NUM_DATA_USED[mode])



