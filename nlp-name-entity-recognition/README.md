# Name Entity Recognition using PyTorch
---

__Just simple example for NER using pytorch__

Named Entity Recognition (NER) is a subtask of Information Extraction that seeks to locate and classify named entities mentioned in unstructured text into predefined categories such as:
1. Person names
2. Organizations
3. Locations
4. Dates and times
5. Numerical values (e.g., money, percentages)
6. Others, depending on the domain

NER is commonly used in:
- Search engines
- Chatbots
- Customer feedback analysis
- Document classification

## PyTorch for NER (Named Entity Recognition)

<div style="text-align: justify;"> PyTorch is an open-source deep learning framework developed by Facebook's AI Research lab (FAIR). It is widely adopted in both academia and industry due to its flexibility, dynamic computation graph, and strong support for GPU acceleration. PyTorch is well-suited for building and training advanced NLP models, including Named Entity Recognition (NER) systems. </div> <br> NER with PyTorch typically involves deep learning architectures such as BiLSTM-CRF or transformer-based models like BERT. Leveraging libraries like Hugging Face Transformers, users can easily access pretrained NER models or fine-tune custom models for domain-specific tasks. PyTorch’s modular design also allows for building scalable and highly customizable NER pipelines.
Key Features:
1. Access to pretrained models like BERT, RoBERTa, DistilBERT, etc.
2. Flexible architecture support: BiLSTM, CRF, or transformer-based models
3. Seamless integration with Hugging Face Transformers for state-of-the-art NLP
4. GPU acceleration and support for distributed training
5. Suitable for both research experimentation and production-level deployment

## Example

__Pre-requisite__
1. Windows
2. Jupyter Notebook or your favorite notebook
2. Python version 3.13.5

First we need to install all of this library in your python environment:
```
pip install ipywidgets ipykernel huggingface_hub[hf_xet] torch torchvision torchaudio transformers
```

__Caution: It will install your latest version__

version list:
1. torch 2.8.0
2. torchvision 0.23.0
3. torchaudio 2.8.0
4. transformers 4.55.0
5. ipywidgets 8.1.7
6. ipykernel 6.29.5
7. huggingface_hub[hf_xet] 0.34.3

So for secure installation:
```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 transformers==4.55.0 ipywidgets==8.1.7 ipykernel==6.29.5 "huggingface_hub[hf_xet]==0.34.3"
```

In my file [ner-pytorch.ipynb](https://github.com/MuhammadMukhlis220/pytorch/blob/main/nlp-name-entity-recognition/ner-pytorch.ipynb) it used pretrained model downloaded from [HuggingFace](https://huggingface.co/). It will download automatically to your device in default directory (Windows) `C:\Users\<your user name>\.cache\huggingface`

__Optional: Set directory for model downloaded__
```bash
conda env config vars set TORCH_HOME="C:\Users\<your user name>\Desktop\project\ds\pytorch\model"
conda env config vars set TRANSFORMERS_CACHE="C:\Users\<your user name>\Desktop\project\ds\pytorch\model"
```
Here example figure when your device is downloading model:

![Alt Text](https://github.com/MuhammadMukhlis220/pytorch/blob/main/nlp-name-entity-recognition/pic/initiate_model_download.png)
Figure 1

We will give entity results to our string from `Natus Vincere is an ukrainian esports organization based in Kyiv, Ukraine. It was founded in 2009 by Alexander Kokhanovsky and Yevhen Zolotarov. The organization is known for its Counter-Strike: Global Offensive team, which has won multiple championships`.

And here the result:

![Alt Text](https://github.com/MuhammadMukhlis220/pytorch/blob/main/nlp-name-entity-recognition/pic/result_1.png)

Figure 2

__That all, give it a try!__
