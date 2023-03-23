# Finetune T5Encoder for text classification

T5 (Text-To-Text Transfer Transformer) 是一个 text-to-text 模型，可以利用其 Encoder 部分来做 text classification，且具有不错的性能。

## 代码使用流程

1. 运行 `load_dataset.py` 下载并处理数据集并保存到本地，这里使用的是 tweet_eval 数据集中的情感分析数据集 (4 分类) 
关于 tweet_eval，可以通过 [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://aclanthology.org/2020.findings-emnlp.148/) 或 [Hugging Face tweet_eval](https://huggingface.co/datasets/tweet_eval) 了解更多。
2. 一些参数设置在 `params.py`, 可以根据自己需求更改，当然也可以通过在终端输入指令的方式来修改这些参数  
3. 运行 `train.py` 即可开始训练，训练好的模型默认保存在 `weights` 文件夹  
如果您想查看训练过程中各个参数/指标的变化，代码中提供了两种记录方式：  
- 在 `logs` 目录下查看相应的 `txt` 文件，记录了训练所设置的参数以及每个 epoch 的评价指标  
- 在 `runs` 目录下保存的是使用 tensorboard 记录的结果，假设路径为 `runs/t5-base_Mar23_17-37-57`，可以通过 `tensorboard --logdir=runs/t5-base_Mar23_17-37-57` 来启动 tensorboard  
4. 运行 `test.py` 即可评估模型在测试集上的效果
假设我保存的模型权重名称为 `t5-base-Mar23_17-37-57-epoch0.pth`，您可以通过在终端输入以下指令来运行 `test.py`:
```commandline
python test.py --weights_name=t5-base-Mar23_17-37-57-epoch0.pth
``` 
&emsp;&emsp;或者直接在 `test.py` 中将 `parser.add_argument('--weights_name', type=str, default=None)` 改为 `parser.add_argument('--weights_name', type=str, default="t5-base-Mar23_17-37-57-epoch0.pth")`

## 使用自己的数据集进行 fine-tune
按照 [text_classification/datasets/tweet_eval-emotion_processed](https://github.com/friedrichor/HuggingFace-Tutorial/tree/main/text_classification/datasets/tweet_eval-emotion_processed) 中的数据格式，把你的数据集也处理成这种格式 (json文件) 即可:
```commandline
[
    {
        "text": ..., 
        "label": ...
    }, 
    ..., 
    {
        "text": ..., 
        "label": ...
    }
]
```
