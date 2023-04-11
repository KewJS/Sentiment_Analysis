import re
import os
import numpy as np
import pandas as pd
import contractions
from typing import Any, List, Dict
from sklearn.model_selection import train_test_split

import nltk
from transformers import BertTokenizer

from src.config import Config


class Logger():
    info = print
    warnings = print
    critical = print
    error = print
    

class Train(Config):
    data = {}
    
    def __init__(self, logger=Logger()):
        self.logger = logger
        
    
    def prepare_model_data(self):
        self.logger.info("Using Reviews Data For Model Training on Sentiment Analysis:")
        
        self.logger.info("  reading reviews data...")
        fname = os.path.join(self.FILES["PREPROCESS_DIR"], self.FILES["REVIEWS_ABT_FILE"])
        self.data["reviews_abt"] = pd.read_parquet("{}.parquet".format(fname))
        
        self.logger.info("  drop those rows with non-missing rating...")
        self.data["reviews_abt"] = self.data["reviews_abt"][self.data["reviews_abt"]["rating"].notnull()]
        
        self.logger.info("  encode sentiments to rating...")
        self.data["sentiment"] = self.data["reviews_abt"]["rating_encode"].apply(lambda x: self.sentiment_encoder(x))
        
        self.logger.info("  initiate tokenizer model: {}".format(self.MODELLING_CONFIG["PRE_TRAINED_MODEL_NAME"]))
        self.tokenizer = BertTokenizer.from_pretrained(self.MODELLING_CONFIG["PRE_TRAINED_MODEL_NAME"])
        
        self.logger.info("  get the token lengths...")
        self.token_lens = []
        for txt in self.data["reviews_abt"]["reviews"]:
            tokens = self.tokenizer.encode(txt, max_length=self.MODELLING_CONFIG["MAX_LEN"], truncation=self.MODELLING_CONFIG["MAX_LEN_TRUNCATION"])
            self.token_lens.append(len(tokens))
        
        self.logger.info("  creating train, test data on ratio & test, validate data on ratio {}...".format(self.MODELLING_CONFIG["TEST_SPLIT_RATIO"], self.MODELLING_CONFIG["VALIDATE_SPLIT_RATIO"]))
        self.data["train_df"], self.data["test_df"] = train_test_split(self.data["reviews_abt"], test_size=self.MODELLING_CONFIG["TEST_SPLIT_RATIO"], random_state=self.MODELLING_CONFIG["RANDOM_SEED"])
        self.data["val_df"], self.data["test_df"] = train_test_split(self.data["test_df"], test_size=self.MODELLING_CONFIG["VALIDATE_SPLIT_RATIO"], random_state=self.MODELLING_CONFIG["RANDOM_SEED"])
        
        if self.QDEBUG:
            fname = os.path.join(self.FILES["PREPROCESS_DIR"], "{}.parquet".format(self.FILES["TRAIN_ABT"]))
            self.data["train_df"].to_parquet(fname)
            
            fname = os.path.join(self.FILES["PREPROCESS_DIR"], "{}.parquet".format(self.FILES["TEST_ABT"]))
            self.data["test_df"].to_parquet(fname)
            
            fname = os.path.join(self.FILES["PREPROCESS_DIR"], "{}.parquet".format(self.FILES["VAL_ABT"]))
            self.data["val_df"].to_parquet(fname)
        
        
    def sentiment_encoder(self, rating):
        if rating == "positive":
            return 0
        elif rating == "neutral":
            return 1
        elif rating == "negative":
            return 2
        else: 
            return np.nan
    
    
    def add_preprocessed_text(self, data, column, lst_regex=None, punkt=False, lower=False, slang=False, lst_stopwords=None, stemm=False, lemm=False, remove_na=True):
        self.logger.info("  perform final processing on reviews...")
        dtf = data.copy()
        dtf = dtf[ pd.notnull(dtf[column]) ]
        dtf[column+"_clean"] = dtf[column].apply(lambda x: self._utils_preprocess_text(x, lst_regex, punkt, lower, slang, lst_stopwords, stemm, lemm))
        
        dtf["check"] = dtf[column+"_clean"].apply(lambda x: len(x))
        if dtf["check"].min() == 0:
            if remove_na is True:
                dtf = dtf[dtf["check"]>0] 
                
        return dtf.drop("check", axis=1)
        
        
    def _utils_preprocess_text(self, txt, lst_regex=None, punkt=True, lower=True, slang=True, lst_stopwords=None, stemm=False, lemm=True):
        if lst_regex is not None: 
            for regex in lst_regex:
                txt = re.sub(regex, "", txt)

        self.logger.info("    > separate sentences with '. '")
        txt = re.sub(r"\.(?=[^ \W\d])", ". ", str(txt))
        
        self.logger.info("    > remove punctuations and characters")
        txt = re.sub(r"[^\w\s]", "", txt) if punkt is True else txt
        
        self.logger.info("    > strip")
        txt = " ".join([word.strip() for word in txt.split()])
        
        self.logger.info("    > lowercase")
        txt = txt.lower() if lower is True else txt
        
        self.logger.info("    > slang")
        txt = contractions.fix(txt) if slang is True else txt
        
        self.logger.info("    > tokenize (convert from string to list)")
        lst_txt = txt.split()
        
        self.logger.info("    > stemming (remove -ing, -ly, ...)")
        if stemm is True:
            ps = nltk.stem.porter.PorterStemmer()
            lst_txt = [ps.stem(word) for word in lst_txt]

        self.logger.info("    > lemmatization (convert the word into root word)")
        if lemm is True:
            lem = nltk.stem.wordnet.WordNetLemmatizer()
            lst_txt = [lem.lemmatize(word) for word in lst_txt]

        self.logger.info("    > stopwords")
        if lst_stopwords is not None:
            lst_txt = [word for word in lst_txt if word not in lst_stopwords]
        
        self.logger.info("    > back to string")
        txt = " ".join(lst_txt)
        
        return txt
    
    
    def _create_stopwords(self, lst_langs=["english"], lst_add_words=[], lst_keep_words=[]):
        lst_stopwords = set()
        for lang in lst_langs:
            lst_stopwords = lst_stopwords.union(set(nltk.corpus.stopwords.words(lang)))
        lst_stopwords = lst_stopwords.union(lst_add_words)
        lst_stopwords = list(set(lst_stopwords) - set(lst_keep_words))
        
        return sorted(list(set(lst_stopwords)))
    
    
    def generate_tokenized_features(self, input_df:pd.DataFrame):        
        tokenized_data = self.preprocess_function(input_df)
        input_df["input_ids"] = tokenized_data["input_ids"]
        input_df["attention_mask"] = tokenized_data["attention_mask"]
        input_df["labels"] = tokenized_data["labels"]

        tokenized_data_dict = input_df.to_dict(orient="list")
        tokenized_features_dict = input_df[["input_ids", "attention_mask", "labels"]].to_dict(orient="index")
        
        return input_df, tokenized_data_dict, tokenized_features_dict
    
    
    @staticmethod
    def histogram_plot(xvar, xlabel, title=None):
        fig, ax = plt.subplots(figsize=(15,4))
        sns.histplot(x=xvar, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontisize=12, weight="bold")
        
        return fig