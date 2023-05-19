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
    :param mode: Choose from ["train", "validation", "test"]. For file naming
    :return: None
    """
    print(f"There are {len(data)} pieces of data in the {mode} set")

    data_dic_list = []
    for per_data in tqdm(data):
        data_dic_list.append({"text": per_data["text"], "label": per_data["label"]})

    # Writing to a json file
    dataset_folder = os.path.join(sys.path[0], "../dataset")
    os.makedirs(dataset_folder, exist_ok=True)
    with open(os.path.join(dataset_folder, mode + ".json"), 'w', encoding='utf-8') as fout:
        json.dump(data_dic_list, fout, indent=4)


if __name__ == '__main__':
    # 使用 tweet_eval 数据集中的情感分析数据集 (4 分类)
    # https://huggingface.co/datasets/tweet_eval

    # text: a string feature containing the tweet.
    # label: an int classification label with the following mapping:
    #        0: anger
    #        1: joy
    #        2: optimism
    #        3: sadness

    dataset = load_dataset("tweet_eval", "emotion")
    print(dataset)  # 查看数据集格式
    pprint(dataset['train'][0], width=1000)  # 查看数据格式
    print("=" * 100)
    
    for mode in ["train", "validation", "test"]:
        splited_data = dataset[mode]
        process_data(splited_data, mode)
