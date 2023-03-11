import os
import sys
import json
from tqdm import tqdm

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

from params import args
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


def main():
    # tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer, num_add_token = tokenizer_plus(tokenizer)

    # config
    config = GPT2Config.from_pretrained(args.model_name)
    config.vocab_size += num_add_token

    # load model weights
    weights = os.path.join(sys.path[0], "weights", "DialoGPT-small-Mar10_20-19-24-epoch2-ppl1.757.pth")  # 这里需要改成你训练后保存模型的路径
    model = GPT2LMHeadModel.from_pretrained(weights, config=config).to(args.device)
    model.eval()

    # data
    test_data_file = os.path.join(args.data_dir, "test_evaluate.json")
    test_data = read_json(test_data_file)

    # save results
    result_folder = os.path.join(sys.path[0], 'results')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    response_save_path = os.path.join(result_folder, 'gt_pred_response.json')
    response_dic_list = []

    """
    # metric init
    sum_bleu_1, sum_bleu_2, sum_bleu_3, sum_bleu_4 = 0, 0, 0, 0
    rouger = Rouge()
    sum_rouge_1, sum_rouge_2, sum_rouge_l = 0, 0, 0
    """

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_data)):
            context, gt_response = data["context"], data["response"]
            pred_response = generate_response(context, tokenizer, model, args.device)

            response_dic_list.append({"ground_truth": gt_response, "predicted": pred_response})

            """
            gt_response_words = gt_response.split(' ')
            pred_response_words = pred_response.split(' ')

            # bleu
            sum_bleu_1 += sentence_bleu([gt_response_words], pred_response_words, weights=(1, 0, 0, 0))
            sum_bleu_2 += sentence_bleu([gt_response_words], pred_response_words, weights=(0.5, 0.5, 0, 0))
            sum_bleu_3 += sentence_bleu([gt_response_words], pred_response_words, weights=(0.33, 0.33, 0.33, 0))
            sum_bleu_4 += sentence_bleu([gt_response_words], pred_response_words, weights=(0.25, 0.25, 0.25, 0.25))

            # rouge
            if pred_response != "":  # 若pred_response为空，则计算rouge时会报错
                rouge = rouger.get_scores(hyps=pred_response, refs=response)[0]
                sum_rouge_1 += rouge['rouge-1']['f']
                sum_rouge_2 += rouge['rouge-2']['f']
                sum_rouge_l += rouge['rouge-l']['f']

            print(f"BLEU-1 = {sum_bleu_1/(i+1)*100}, BLEU-2 = {sum_bleu_2/(i+1)*100}, BLEU-3 = {sum_bleu_3/(i+1)*100}, BLEU-4 = {sum_bleu_4/(i+1)*100}")
            print(f"ROUGE-1 = {sum_rouge_1/(i+1)*100}, ROUGE-2 = {sum_rouge_2/(i+1)*100}, ROUGE-l = {sum_rouge_l/(i+1)*100}")
            """

    # 写入json文件
    fout_path = os.path.join(response_save_path)
    with open(fout_path, 'w', encoding='utf-8') as fout:
        json.dump(response_dic_list, fout, indent=4)


if __name__ == "__main__":
    main()