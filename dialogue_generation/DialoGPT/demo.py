import os
import sys
import json
from tqdm import tqdm
import argparse

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

import params
from utils import tokenizer_plus, read_json

import warnings
warnings.filterwarnings("ignore")


def generate_response(context: str, tokenizer, model, device):
    input_ids = tokenizer.encode(context + tokenizer.eos_token,
                                 add_special_tokens=False,
                                 return_tensors='pt')

    # num_beams=1且do_sample=False 即 greedy search
    generated_ids = model.generate(input_ids.to(device), num_beams=1, do_sample=False,
                                   max_new_tokens=128, pad_token_id=tokenizer.pad_token_id)

    response = tokenizer.decode(generated_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response


def main(args):
    # tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer, num_add_token = tokenizer_plus(tokenizer)

    # config
    config = GPT2Config.from_pretrained(args.model_name)
    config.vocab_size += num_add_token

    # load model weights
    if args.weights_name == None:
        raise ValueError("weights_name is not available. Please set the weights_name")
    weights = os.path.join(args.weights_dir, args.weights_name)
    model = GPT2LMHeadModel.from_pretrained(weights, config=config).to(params.device)
    model.eval()

    for _ in range(5):
        user_input = input(">> User:")
        response = generate_response(user_input, tokenizer, model, args.device)

        print('Bot:', response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    TEXT_DIALOGUE_MODEL = ['microsoft/DialoGPT-small', 'microsoft/DialoGPT-medium', 'microsoft/DialoGPT-large']
    parser.add_argument('--model_name', type=str, choices=TEXT_DIALOGUE_MODEL, default=params.model_name)
    parser.add_argument('--tokenizer_name', type=str, default=params.tokenizer_name)

    parser.add_argument('--device', default=params.device)
    parser.add_argument('--nw', type=int, default=params.num_workers)
    parser.add_argument('--data_dir', type=str, default=params.data_dir)
    parser.add_argument('--weights_dir', type=str, default=params.weights_dir)
    parser.add_argument('--weights_name', type=str, default=None)

    args = parser.parse_args()

    main(args)