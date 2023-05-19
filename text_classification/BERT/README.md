# Fine-tune BERT for text classification

## 数据集

&emsp;&emsp;使用 tweet_eval emotion 数据集 (情感分析，4分类)，其中训练集 3257, 验证集 374, 测试集 1421)  

&emsp;&emsp;可以通过 [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://aclanthology.org/2020.findings-emnlp.148/) 或 [Hugging Face tweet_eval](https://huggingface.co/datasets/tweet_eval) 了解更多

运行 `load_and_process_dataset/tweet_eval.py`。下载并处理 tweet_eval emotion 数据集到本地，处理成：
```json
[
    {
        "text": "\u201cWorry is a down payment on a problem you may never have'. \u00a0Joyce Meyer.  #motivation #leadership #worry",
        "label": 2
    },
    {
        "text": "My roommate: it's okay that we can't spell because we have autocorrect. #terrible #firstworldprobs",
        "label": 0
    },
    ...
]
```
运行后的结果可以在 [text_classification/BERT/dataset](https://github.com/friedrichor/NLP-HuggingFace-Tutorial/tree/main/text_classification/BERT/dataset) 找到 (若因网络等原因无法连接到 Hugging Face 导致运行报错，可以直接使用该文件)

<hr>

## 硬件设备要求

若使用 `bert-base-uncased` 作为预训练模型：
- 若设置 `train_batch_size=64`，需要一块 8GB 的 GPU


<hr>

## 训练相关参数和路径等

&emsp;&emsp;在 `params.py` 可以找到默认值，可以在该文件进行修改和设置。  

&emsp;&emsp;若您了解如何使用 shell 脚本运行代码，也可以通过 `train.sh` 设置相关参数和路径  

<hr>

## 运行

### **下载并处理 ProsocialDialog 数据集**


```commandline
git clone https://github.com/friedrichor/NLP-HuggingFace-Tutorial
cd NLP-HuggingFace-Tutorial/text_classification/BERT
python load_and_process_dataset/tweet_eval.py
```

### **训练**
```commandline
python train.py
```
或
```commandline
sh train.sh
```

&emsp;&emsp;此外，训练时使用了 logging 和 tensorboard 记录训练时各个参数/指标的变化。关于 tensorboard，可以通过以下命令查看
```
tensorboard --logdir=runs
```

### **测试**


若训练后保存模型参数为 `weights/bert-base-uncased-May19_23-32-00-epoch5-macro_f10.709.pth` 且使用该模型用于测试，则可通过以下代码测试
```
python test.py --weights_name bert-base-uncased-May19_23-32-00-epoch5-macro_f10.709.pth
```

<hr>

## 使用自己的数据集进行训练
按照 `dialogue_generation/DialoGPT/dataset` 或 `dialogue_generation/DialoGPT/partial_dataset` 中的格式，将自己的数据集改成相同格式，即
```json
[
    {
        "text": "\u201cWorry is a down payment on a problem you may never have'. \u00a0Joyce Meyer.  #motivation #leadership #worry",
        "label": 2
    },
    {
        "text": "My roommate: it's okay that we can't spell because we have autocorrect. #terrible #firstworldprobs",
        "label": 0
    },
    ...
]
```
