# Finance_Chatbot
<p align="center"><img width="1000" height="300" src="https://editor.analyticsvidhya.com/uploads/30783expressanlytics.jpg"></p>

This project initiated from **Sentiment Analysis**, using ensemble model like XGBoost and advance language model like BERT with  words embedding to build a sentiment analysis application, focusing on understand the sentiments from reviews data. Reviews data were scraped from online resources, and then these reviews contents with its ratings will be used as input for **Sentiment Analysis**. Due to the size that the data are big, hence, they are not uploaded into this GitHub repository.

## Table of Contents
* **1. About the Project**
* **2. Getting Started**
* **3. Set up your environment**
* **4. Open your Jupyter notebook**


## Structuring a repository
An integral part of having reusable code is having a sensible repository structure. That is, which files do we have and how do we organise them.
- Folder layout:
```bash
customer_segmentation
├── src
|   └── static
|       └── img
|         └── sentiments.jpg
|   └── templates
|       └── sentiment_analysis.md
|   └── preprocess
|       └── __init__.py
|   └── spider
|       └── __init__.py
|       └── crawler.py
|   └── train
|       └── __init__.py
|       └── bert.py
|       └── roberta.py
|       └── models.py
|   └── config.py
|   └── app.py
├── .gitignore
├── README.md
├── requirements.txt
├── Analysis.ipynb
├── BERT_Train.ipynb
├── CNN_Train.ipynb
├── XGBoost_Train.ipynb
└── Scraper.ipynb
```


## 1. About the Project
With this reviews data - <font color='blue'>reviews_abt.jparquetson</font>, let kick started on it:
  - <b><u>Scrape online reviews data</u></b>
  - <b><u>Preprocess the text information given in the reviews data like stemming, removing stopwords, lematization...</u></b>
  - <b><u>Word embedding, creating words vectors using techniques like Bags of Words (BOW)</u></b>
  - <b><u>Create ensembles model like XGBoost for sentiment analysis</u></b>
  - <b><u>Create deep learning & language models like CNN and LSTM BERT (work in progress) for sentiment analysis</u></b>
  - <b><u>Evaluate the models performance on sentiment analysis</u></b>