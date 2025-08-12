https://huggingface.co/blog/sentiment-analysis-python

# Sentiment Analysis using PyTorch (CPU Mode)
---

__Just simple example for SA using pytorch__

To anyone who want get complete version from this repo, you can go to this [link](https://huggingface.co/blog/sentiment-analysis-python). It provides the basic of sentiment analysis and code that i already copy paste for this repo).

## Example

__Pre-requisite__
1. Windows
2. Jupyter Notebook or your favorite notebook
2. Python version 3.13.5

Library list:
1. torch 2.8.0
2. torchvision 0.23.0
3. torchaudio 2.8.0
4. transformers 4.55.0
5. ipywidgets 8.1.7
6. ipykernel 6.29.5
7. huggingface_hub[hf_xet] 0.34.3

Install with:
```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 transformers==4.55.0 ipywidgets==8.1.7 ipykernel==6.29.5 "huggingface_hub[hf_xet]==0.34.3"
```

Simple code just like from Hugging Face's doc, we only need pipeline with parameter __task__ is `sentiment-analysis`. By default it will use default model: `distilbert-base-uncased-finetuned-sst-2-english`

Here the log when the model is not mentioned:
![Alt Text](https://github.com/MuhammadMukhlis220/pytorch/blob/main/nlp-sentiment-analysis/pic/initiate_model_1.png)
Figure 1

From file pipeline.py:
<details>
   <summary>Click to show default pipeline config for sentiment-analysis</summary>

   ```python
"sentiment-analysis": {
    "impl": TextClassificationPipeline,
    "tf": TFAutoModelForSequenceClassification if is_tf_available() else None,
    "pt": AutoModelForSequenceClassification if is_torch_available() else None,
    "default": {
        "model": {
            "pt": "distilbert-base-uncased-finetuned-sst-2-english",
            "tf": "distilbert-base-uncased-finetuned-sst-2-english",
        },
    },
},

   ```
   </details>

In our file [ner-pytorch.ipynb](https://github.com/MuhammadMukhlis220/pytorch/blob/main/nlp-sentiment-analysis/sentiment-analysis-pytorch.ipynb) it consist 2 language example of comments, Indonesia and English. This model is capable for that 2 languages.

Here the results:
![Alt Text](https://github.com/MuhammadMukhlis220/pytorch/blob/main/nlp-sentiment-analysis/pic/result_1.png)
Figure 2

If you don't want use default model and want to use your favorite model, you can use this code:

<details>
   <summary>Click to show for full code</summary>

   ```python

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english" # Change in here for your favorite model

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

comments = [
    "Gokil nih, keren banget!",
    "Gak ngerti deh sama yang ini, bikin pusing",
    "Wkwk, lucu parah sih chatnya",
    "b aja sih, gak sebagus yang diharap",
    "terimakasih ya bro!",
    "i dont think this is good idea",
    "I love this! Soooo amazing.",
    "This is the worst experience I've ever had"
]

results = sentiment_pipeline(comments)

for text, result in zip(comments, results):
    print(f"{text} -> {result}")

   ```
   </details>


