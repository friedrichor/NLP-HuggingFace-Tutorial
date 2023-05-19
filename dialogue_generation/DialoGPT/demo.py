import os
import argparse

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

import params

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

    with torch.no_grad():
        for _ in range(5):
            user_input = input(">> User:")
            input_ids = tokenizer(user_input,
                                  add_special_tokens=True,
                                  return_tensors='pt').input_ids
            # num_beams=1 & do_sample=False -> greedy search
            # num_beams>1 & do_sample=False -> beam search
            outputs = model.generate(input_ids.to(args.device),
                                     num_beams=5, do_sample=False,
                                     max_new_tokens=128,
                                     pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
            print('Bot:', response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model_name_or_path', type=str, default=params.pretrained_model_name_or_path)

    parser.add_argument('--device', default=params.device)
    parser.add_argument('--weights_dir', type=str, default=params.weights_dir)
    parser.add_argument('--weights_name', type=str, default=None)

    args = parser.parse_args()

    main(args)
