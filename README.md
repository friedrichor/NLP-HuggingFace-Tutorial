# NLP-HuggingFace-Tutorial

使用 Hugging Face 以及 PyTorch 做一些 NLP 任务的 tutorial

代码相对简洁易懂，适合新手入门，还在更新中……

如有任何问题，可以在 [issues](https://github.com/friedrichor/NLP-HuggingFace-Tutorial/issues) 中提出，也可以在 [CSDN/friedrichor](https://blog.csdn.net/Friedrichor) 私信我。

## 对话生成

备注：对话生成的代码同样适用于文本生成、文本摘要等 text-to-text 任务。

- DialoGPT
  - paper: [DIALOGPT : Large-Scale Generative Pre-training for Conversational Response Generation](https://aclanthology.org/2020.acl-demos.30/)  (ACL 2020)  
  - fine-tune DialoGPT for dialogue generation: [code](https://github.com/friedrichor/NLP-HuggingFace-Tutorial/tree/main/dialogue_generation/DialoGPT)
- T5
  - paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html) (JMLR 2020)
  - fine-tune T5 for dialogue generation: [code](https://github.com/friedrichor/NLP-HuggingFace-Tutorial/tree/main/dialogue_generation/T5)

## 文本分类

备注：情感分析、主题分类、意图识别等任务均属于文本分类。

- BERT
  - paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) (NAACL 2019)
  - fine-tune BERT for text classification: [code](https://github.com/friedrichor/NLP-HuggingFace-Tutorial/tree/main/text_classification/BERT)
  - prompt-tuning BERT for text classification: [code](https://github.com/friedrichor/NLP-HuggingFace-Tutorial/tree/main/text_classification_Prompt/BERT)
- RoBERTa
  - paper: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) (arXiv 2019)
  - fine-tune RoBERTa for dialogue generation: [code](https://github.com/friedrichor/NLP-HuggingFace-Tutorial/tree/main/text_classification/RoBERTa)
- T5
  - paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html) (JMLR 2020)
  - fine-tune T5 for text classification: [code](https://github.com/friedrichor/NLP-HuggingFace-Tutorial/tree/main/text_classification/T5)
  - fine-tune T5 Encoder for text classification: [code](https://github.com/friedrichor/NLP-HuggingFace-Tutorial/tree/main/text_classification/T5Encoder)

更新中...
