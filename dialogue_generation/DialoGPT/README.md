# Fine-tune DialoGPT

## 代码使用流程：
1. 运行 `load_dataset.py` 下载并处理数据集并保存到本地，这里使用的是 ProsocialDialog 数据集 (训练集120236, 验证集20416, 测试集25029)，文档：[https://huggingface.co/datasets/allenai/prosocial-dialog](https://huggingface.co/datasets/allenai/prosocial-dialog)  
如果fine-tune DialoGPT-small, batch_size=8, 显存8G就能跑全部数据。  
如果显存不够的话或者只想简单跑一下看能不能运行，可以运行 `load_partial_dataset.py`，这里我只截取了 4000 个训练集数据，验证集和测试集数据均500个  
如果显存还不够的话可以截取更少的训练集，如 2000个、1000个，也可以把 batch_size 改小，如4，不过更少的训练集可能导致模型效果不是很好  
2. 训练相关参数都在 `params.py`, 根据自己需求更改即可    
如果只用部分数据集训练，设置 `data_dir = os.path.join(sys.path[0], 'dataset_partial')`  
如果用全部数据集训练，设置 `data_dir = os.path.join(sys.path[0], 'dataset')`  
3. 运行 `train.py` 即可开始训练，训练好的模型默认保存在 `weights` 文件夹
4. 运行 `generate.py` 生成测试集的回复，用于后续评价模型性能  
运行后会将 ground-truth 的 response 和fine-tune后模型预测出的 response 保存在 `results/gt_pred_response.json`  
其中第40行代码 `weights = os.path.join(args.weights_dir, "DialoGPT-small-Mar10_20-19-24-epoch2-ppl1.757.pth")` 需要把这个文件名改成你自己训练好的模型名称  
5. 运行 `evaluate.py` 提供了计算 BLEU 和 ROUGE 评价指标的代码，用于评估模型性能

## 其他说明
1. ProsocialDialog 是一个单轮对话的数据集，运行`load_dataset.py`后的结果放在了`dialogue_generation/datasets/ProsocialDialog_processed`  
如果想用你自己的数据训练，把数据处理成和我的数据格式相同即可，训练代码无需更改  
如果是用多轮对话的数据集可以参考我处理后的数据格式，如 `sentence1 [SEP] sentence2 [SEP] sentence3 [SEP] sentence4 [EOS]`, 
这里的 `[SEP]` 和 `[EOS]` 分别指 `sep_token` 和 `eos_token`。各个Tokenizer中 `sep_token` 和 `eos_token` 的定义还不太一样，如 BERT 中的 `sep_token` 就是 `[SEP]`，`eos_token` 就是 `[EOS]`，
而 DialoGPT 的 tokenizer 并没有定义 `sep_token` (需要用的话要用 `add_special_tokens`)，且 `eos_token` 为 `<|endoftext|>`。这里我也只是举个例子，提供个参考，实际以你自己做的任务为准。
2. 训练时使用了 logging 和 tensorboard 记录训练时各个参数/指标的变化
3. `generate.py` 中生成回复时使用的策略为greedy search，可以根据自己需要改进，这里只是举个简单的例子
4. 在计算评价指标时，并没有用 huggingface 中的 evaluate 库，因为 `evaluate.load` 加载计算评价指标时太过漫长，所以用的是 `nltk` 和 `rouge` 库分别计算 BLEU 和 ROUGE
