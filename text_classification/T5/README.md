# Finetune T5 for text classification

T5 (Text-To-Text Transfer Transformer) 是一个 text-to-text 模型，可以直接用来做 text classification。  
其性能甚至要比一些专门做分类的模型还要好，例如本代码所使用的 tweet_eval 数据集，很轻松就超过 PapersWithCode 中的 SOTA 模型。

## 硬件要求
对于 `t5-base`:
- 若 `--use_Adafactor=False`
  - 若 batch size=16，需要一块 8G 的 GPU；
  - 若 batch size=8，需要一块 6G 的 GPU。
- 若 `--use_Adafactor=True`
  - 若 batch size=16，需要一块 8G 的 GPU (只占用 7G 显存)；
  - 若 batch size=8，需要一块 4G 的 GPU。

## 代码使用流程

先安装 sentencepiece:  
```
pip install sentencepiece
```

1. 运行 `load_dataset.py` 下载并处理数据集并保存到本地，这里使用的是 tweet_eval 数据集中的情感分析数据集 (4 分类) 
关于 tweet_eval，可以通过 [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://aclanthology.org/2020.findings-emnlp.148/) 或 [Hugging Face tweet_eval](https://huggingface.co/datasets/tweet_eval) 了解更多。
2. 一些参数设置在 `params.py`, 可以根据自己需求更改，当然也可以通过在终端输入指令的方式来修改这些参数  
其中 `prefix_text` 是 fine-tune T5 时所使用的前缀文本，根据自己需要更改，具体可查看 T5 论文中的附录部分  
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

## Adafactor

paper: [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](http://proceedings.mlr.press/v80/shazeer18a.html)  

&emsp;&emsp;T5 作者推荐使用 Adafactor 来替代传统的 Adam 系列优化器：  
- 训练时 Adam 会占用大量的显存，尤其像 t5-3b 这种参数量比较大的模型，如果数据量也很大的话，训练时会占用极大的显存，对于 GPU 的要求很高
- 作者在论文中说 Adafactor 是专门为 Transformer/T5 所设计的，可以有效减少显存占用。

&emsp;&emsp;在 `train.py` 中可以通过 `--use_Adafactor` 和 `--use_AdafactorSchedule` 来设置是否使用 Adafactor。  
&emsp;&emsp;HuggingFace 文档中的说明: [transformers.Adafactor](https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/optimizer_schedules#transformers.Adafactor)

## 使用自己的数据集进行 fine-tune
按照 [text_classification/datasets/tweet_eval-emotion_processed](https://github.com/friedrichor/NLP-HuggingFace-Tutorial/tree/main/text_classification/datasets/tweet_eval-emotion_processed) 中的数据格式，把你的数据集也处理成这种格式 (json文件) 即可:
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
2. 按照 [text_classification/T5/classes_map.json](https://github.com/friedrichor/NLP-HuggingFace-Tutorial/tree/main/text_classification/T5/classes_map.json) 中的格式，把你的数据集也处理成这种格式 (json文件) 即可:
```commandline
{
    "class_1": 0,
    "class_2": 1,
    ...
    "class_n": n-1
}
```

## 数据集样本分布极不均衡

若您的数据集样本分布极不均衡，可以通过 `--use_weighted_random_sampler` 来设置是否使用 [WeightedRandomSampler](https://pytorch.org/docs/stable/data.html?highlight=weightedrandomsampler#torch.utils.data.WeightedRandomSampler)  
关于 WeightedRandomSampler，可以参考我的博客 [使用 WeightedRandomSampler 解决数据样本不均衡的问题](https://blog.csdn.net/Friedrichor/article/details/129901346)  
