# README

**Some Popular Text Classification Models with Toxic Text Classification**

In this project, I explore some popular text classification models:

- Latent Semantic Analysis 
- Word2Vec and Logistic Regression 
- Fine-tuning BERT
- LORA fine-tuning LLAMA 3.2 1B

On dataset:  [Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) 

### How to Run

- Place the data (extracted csv files) in a data folder in the root node, also with the process_data.py file.
- Run the jupyter notebooks, notice you need permission to download LLAMA model from transformers.

### Data Preprocess

Due to the limitation of computation resources and concerns about model performance, I did following prepocess:

- Filter out non-English comments
- Only choose to predict "toxic" label
- Only use the smaller train data and splitted validation set
- Balanced positive and negative by downsampling negative data

### Environment & Packages

The key packages used for the models

- LSA: sklearn
- Word2Vec: nltk, gensim, sklearn
- BERT: pytorch, transformers
- LLAMA: pytorch, transformers, peft

### Hardware for Running

- CPU: Ryzen 5 2600
- RAM: 32G
- GPU: RTX 3060 - 12G
- OS: Pop-OS

### Brief Result

Fine-tuning LLM with LORA on local machine is practical and produces good performance.
