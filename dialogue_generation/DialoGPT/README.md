# Fine-tune DialoGPT for Text Dialogue

## 数据集

&emsp;使用 ProsocialDialog 数据集 (单轮对话数据集)，其中训练集 120,236, 验证集 20,416, 测试集 25,029)  

&emsp;可从 Hugging Face ([https://huggingface.co/datasets/allenai/prosocial-dialog](https://huggingface.co/datasets/allenai/prosocial-dialog)) 了解更多   

运行 `load_and_process_dataset/prosocial_dialog.py`。下载并处理 ProsocialDialog 数据集到本地，处理成：
```json
[
    {
        "dialogue": "I tried a bit of shadowboxing today. Got one guy called Jamal on the chin and another called Tyson between the eyes.<|endoftext|> This isn't a joke about black people, right? If it is, it isn't funny.<|endoftext|>"
    },
    {
        "dialogue": "You're right, it isn't funny. Finding enjoyment in other people's pains isn't funny.<|endoftext|> I am glad that you agree. Joking about abusing black people can quickly get you marked as a racist.<|endoftext|>"
    },
    ...
]
```
运行后的结果可以在 [dialogue_generation/DialoGPT/dataset](https://github.com/friedrichor/NLP-HuggingFace-Tutorial/tree/main/dialogue_generation/DialoGPT/dataset) 找到 (若因网络等原因无法连接到 Hugging Face 导致运行报错，可以直接使用该文件)

<hr>

## 硬件设备要求

若使用 `DialoGPT-small` 作为预训练模型：
- 当使用全部数据用于训练时
  - 若设置 `train_batch_size=64`，需要一块 48GB 的 GPU，训练一个 epoch 需要约 12min
  - 若设置 `train_batch_size=32`，需要一块 24GB 的 GPU，训练一个 epoch 需要约 12min
  - 若设置 `train_batch_size=8`，需要一块 12GB 的 GPU，训练一个 epoch 需要约 16min
- 当使用部分数据 (从数据集中截取部分样本，训练集4k，验证集4k，测试集500) 用于训练时，若设置 `train_batch_size=8`，需要一块 8GB 的 GPU，训练一个 epoch 仅需 32s  
  <font color=DarkOrange>如果仅仅想测试代码是否可运行，为节省时间，推荐使用部分数据用于训练。  
  可以通过运行 `load_and_process_dataset/partial_prosocial_dialog.py` 获得只有部分训练集和测试集的数据集，运行后的结果可以在 [dialogue_generation/DialoGPT/partial_dataset](https://github.com/friedrichor/NLP-HuggingFace-Tutorial/tree/main/dialogue_generation/DialoGPT/partial_dataset) 找到</font>  

若使用 `microsoft/DialoGPT-medium` 或 `microsoft/DialoGPT-large`，请自行探索所需硬件要求。

<hr>

## 训练相关参数和路径等

&emsp;&emsp;在 `params.py` 可以找到默认值，可以在该文件进行修改和设置。  

&emsp;&emsp;若您了解如何使用 shell 脚本运行代码，也可以通过 `train.sh` 设置相关参数和路径  

&emsp;&emsp;默认设置使用部分数据用于训练和测试，若希望在整个数据集上进行训练和测试，请启用 `data_dir = os.path.join(sys.path[0], "dataset")` (params.py中) 或 `export DATA_DIR="dataset"` (train.sh中)


<hr>

## 运行

### **下载并处理 ProsocialDialog 数据集**


```commandline
git clone https://github.com/friedrichor/NLP-HuggingFace-Tutorial
cd NLP-HuggingFace-Tutorial/dialogue_generation/T5
python load_and_process_dataset/partial_prosocial_dialog.py
```
或
```commandline
git clone https://github.com/friedrichor/NLP-HuggingFace-Tutorial
cd NLP-HuggingFace-Tutorial/dialogue_generation/T5
python load_and_process_dataset/prosocial_dialog.py
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

#### **1. 生成预测回复**

若训练后保存模型参数为 `weights/DialoGPT-small-May19_21-48-12-epoch0-ppl6.219.pth` 且使用该模型用于测试，则可通过以下代码测试
```
python generate_response.py --weights_name DialoGPT-small-May19_21-48-12-epoch0-ppl6.219.pth
```
运行后将会生成文件 `results/gt_pred_response.json`，可用于后续计算评价指标

#### **2. evaluate/计算评价指标**

提供了计算 BLEU 和 ROUGE 评价指标的代码
```
python evaluate.py 
```

### **demo**

使用训练后的模型用于单轮人机对话的 demo (DialoGPT-small-May19_21-48-12-epoch0-ppl6.219.pth 换成自己的模型名)
```
python demo.py --weights_name DialoGPT-small-May19_21-48-12-epoch0-ppl6.219.pth
```

<hr>

## 使用自己的数据集进行训练
按照 `dialogue_generation/DialoGPT/dataset` 或 `dialogue_generation/DialoGPT/partial_dataset` 中的格式，将自己的数据集改成相同格式，即
```json
[
    {
        "dialogue": "I tried a bit of shadowboxing today. Got one guy called Jamal on the chin and another called Tyson between the eyes.<|endoftext|> This isn't a joke about black people, right? If it is, it isn't funny.<|endoftext|>"
    },
    {
        "dialogue": "You're right, it isn't funny. Finding enjoyment in other people's pains isn't funny.<|endoftext|> I am glad that you agree. Joking about abusing black people can quickly get you marked as a racist.<|endoftext|>"
    },
    ...
]
```