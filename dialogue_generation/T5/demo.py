import os
import argparse

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

import params

import warnings
warnings.filterwarnings("ignore")

def main(args):
    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path)
    # config
    config = T5Config.from_pretrained(args.pretrained_model_name_or_path)
    # load model weights
    if args.weights_name == None:
        raise ValueError("weights_name is not available. Please set the weights_name")
    weights = os.path.join(args.weights_dir, args.weights_name)
    model = T5ForConditionalGeneration.from_pretrained(weights, config=config)
    model.to(args.device)

    with torch.no_grad():
        for _ in range(5):
            user_input = input(">> User:")
            input_ids = tokenizer(args.text_prefix + user_input,
                                  add_special_tokens=True,
                                  return_tensors='pt').input_ids
            # num_beams=1 & do_sample=False -> greedy search
            # num_beams>1 & do_sample=False -> beam search
            outputs = model.generate(input_ids.to(args.device),
                                     num_beams=5, do_sample=False,
                                     max_new_tokens=128,
                                     pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print('Bot:', response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model_name_or_path', type=str, default=params.pretrained_model_name_or_path)
    parser.add_argument('--text_prefix', type=str, default=params.text_prefix)

    parser.add_argument('--device', default=params.device)
    parser.add_argument('--weights_dir', type=str, default=params.save_weights_path)
    parser.add_argument('--weights_name', type=str, default=None)

    args = parser.parse_args()

    main(args)
