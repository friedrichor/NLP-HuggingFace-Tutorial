import os
import sys
from typing import List, Dict

from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

from utils import read_json


def compute_bleu(data: List[Dict[str, str]]):
    # metric init
    sum_bleu_1, sum_bleu_2, sum_bleu_3, sum_bleu_4 = 0, 0, 0, 0

    for gt_pred_data in data:
        ground_truth_str = gt_pred_data["ground_truth"]
        predicted_str = gt_pred_data["predicted"]

        ground_truth_words = ground_truth_str.split(' ')
        predicted_words = predicted_str.split(' ')

        # bleu
        sum_bleu_1 += sentence_bleu([ground_truth_words], predicted_words, weights=(1, 0, 0, 0))
        sum_bleu_2 += sentence_bleu([ground_truth_words], predicted_words, weights=(0.5, 0.5, 0, 0))
        sum_bleu_3 += sentence_bleu([ground_truth_words], predicted_words, weights=(0.33, 0.33, 0.33, 0))
        sum_bleu_4 += sentence_bleu([ground_truth_words], predicted_words, weights=(0.25, 0.25, 0.25, 0.25))

    data_size = len(data)
    avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4 = sum_bleu_1 / data_size, sum_bleu_2 / data_size, sum_bleu_3 / data_size, sum_bleu_4 / data_size

    return {  # 一般论文里的 BLEU 单位为 %, 即是乘 100 之后的值
        'BLEU-1': avg_bleu_1 * 100,
        'BLEU-2': avg_bleu_2 * 100,
        'BLEU-3': avg_bleu_3 * 100,
        'BLEU-4': avg_bleu_4 * 100,
    }


def compute_rouge(data: List[Dict[str, str]]):
    # metric init
    rouger = Rouge()
    sum_rouge_1, sum_rouge_2, sum_rouge_l = 0, 0, 0

    for gt_pred_data in data:
        ground_truth_str = gt_pred_data["ground_truth"]
        predicted_str = gt_pred_data["predicted"]

        # rouge
        if predicted_str != "":  # 若pred_response为空，则计算rouge时会报错
            rouge = rouger.get_scores(hyps=predicted_str, refs=ground_truth_str)[0]
            sum_rouge_1 += rouge['rouge-1']['f']
            sum_rouge_2 += rouge['rouge-2']['f']
            sum_rouge_l += rouge['rouge-l']['f']

    data_size = len(data)
    avg_rouge_1, avg_rouge_2, avg_rouge_l = sum_rouge_1 / data_size, sum_rouge_2 / data_size, sum_rouge_l / data_size

    return {  # 一般论文里的 ROUGE 单位为 %, 即是乘 100 之后的值
        'ROUGE-1': avg_rouge_1 * 100,
        'ROUGE-2': avg_rouge_2 * 100,
        'ROUGE-l': avg_rouge_l * 100
    }


if __name__ == "__main__":
    result_folder = os.path.join(sys.path[0], 'results')
    response_file = os.path.join(result_folder, "gt_pred_response.json")
    response_data = read_json(response_file)

    bleu_result = compute_bleu(response_data)
    print(bleu_result)

    rouge_result = compute_rouge(response_data)
    print(rouge_result)




