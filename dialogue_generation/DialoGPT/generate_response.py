import os
import sys
import json
from tqdm import tqdm
import argparse

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

import params
from utils import read_json

import warnings
warnings.filterwarnings("ignore")


def main(args):
    # tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model_name_or_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    # config
    config = GPT2Config.from_pretrained(args.pretrained_model_name_or_path)
    # load model weights
    if args.weights_name == None:
        raise ValueError("weights_name is not available. Please set the weights_name")
    weights = os.path.join(args.weights_dir, args.weights_name)
    model = GPT2LMHeadModel.from_pretrained(weights, config=config)
    model.to(args.device)

    # data
    test_data = read_json(os.path.join(args.data_dir, "test_evaluate.json"))

    # save results
    response_dic_list = []
    
    with torch.no_grad():
        batch_context = []
        batch_gt_response = []
        for i, data in tqdm(enumerate(test_data)):
            # 一次将多个context传给模型生成response，提高生成效率
            batch_context.append(data["context"])
            batch_gt_response.append(data["response"])
            
            if (len(batch_context) == 8) or (i == len(test_data) - 1):
                input_ids = tokenizer(batch_context,
                                      add_special_tokens=True,
                                      padding='max_length',
                                      truncation=True,
                                      max_length=256,
                                      return_tensors='pt').input_ids
                # num_beams=1 & do_sample=False -> greedy search
                # num_beams>1 & do_sample=False -> beam search
                outputs = model.generate(input_ids.to(args.device),
                                         num_beams=5, do_sample=False,
                                         max_new_tokens=128,
                                         pad_token_id=tokenizer.eos_token_id)
                for i, generated_ids in enumerate(outputs):
                    pred_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    response_dic_list.append({"ground_truth": batch_gt_response[i], "predicted": pred_response})
                
                batch_context = []
                batch_gt_response = []
                
    # write in a json file
    result_folder = os.path.join(sys.path[0], 'results')
    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, 'gt_pred_response.json'), 'w', encoding='utf-8') as fout:
        json.dump(response_dic_list, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model_name_or_path', type=str, default=params.pretrained_model_name_or_path)
    parser.add_argument('--data_dir', type=str, default=params.data_dir)

    parser.add_argument('--device', default=params.device)
    parser.add_argument('--weights_dir', type=str, default=params.weights_dir)
    parser.add_argument('--weights_name', type=str, default=None)

    args = parser.parse_args()

    main(args)
