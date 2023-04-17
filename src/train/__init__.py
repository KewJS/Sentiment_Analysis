import re
import os
import sys
import time
import pickle
import getpass
import numpy as np
import pandas as pd
import contractions
from platform import uname
import datetime as dt
from datetime import datetime
from typing import Any, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import nltk
from transformers import BertTokenizer
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, DistilBertModel, DistilBertTokenizer

from src.config import Config
from src.train.ensemble import Model


class Logger():
    info = print
    warnings = print
    critical = print
    error = print
    

class Train(Config):
    data = {}
    
    CLASSFICATION_ALGORITHMS = dict(
        XGBC = dict(alg=XGBClassifier, args=dict(random_state=42, use_label_encoder=False, early_stopping=5, 
                                                enable_categorical=True,
                                                eval_metric="aucpr",
                                                sample_weight=None
                                                )),
        
        LGBMC = dict(alg=LGBMClassifier, args=dict(early_stopping_round=5,
                                                   class_weight=None,
                                                )),
        
        XGBC_TUNED = dict(alg=XGBClassifier, args=dict(random_state=42, use_label_encoder=False, early_stopping=5,
                                                    enable_categorical=True,
                                                    eval_metric="aucpr",
                                                    sample_weight=None),
                        param_grid = {
                            # "scale_pos_weight": [0.5, 1.0, 2.0, 4.5],
                            "n_estimators": [25, 50, 100],
                            "max_delta_step": [0, 1.0, 3.0, 5.0],
                            "max_bin": [2, 5, 7, 10],
                            "max_depth": [5, 6, 8, None],
                            "gamma": [0.5, 1.5, 2.5, 4],
                            "min_child_weight": [0.05, 0.01, 1, 2],
                            "eta": [0.005, 0.01, 0.05],
                            "learning_rate": [0.01, 0.05, 0.1],
                            #  "subsample": [0.5, 0.7],
                            #  "colsample_bytree": [0.5, 0.7],
                            #  "colsample_bylevel": [0.5, 0.7],
                            #  "colsample_bynode": [0.5, 0.7],
                            #  "alpha": [0.5, 0.7, 0.9, 1.3],
                            #  "lambda": [0.5, 0.7, 0.9, 1.3],
                            #  "reg_alpha": [0.5, 0.7, 0.9, 1.3],
                            #  "reg_lambda": [0.5, 0.7, 0.9, 1.3],
                        }
                        ),
        
        LGBMC_TUNED = dict(alg=LGBMClassifier, args=dict(early_stopping_round=5,
                                                         class_weight=None,
                                                    ),
                        param_grid = {
                            # "scale_pos_weight": [0.5, 1.0, 2.0, 4.5],
                            "n_estimators": [100, 500, 1000, 1500],
                            "max_delta_step": [3, 6, 9, 12],
                            "max_bin": [5, 7, 10, 13],
                            "max_depth": [7, 10, 20, 30, 50],
                            "min_child_weight": [0.05, 0.01, 1, 2],
                            "min_sum_hessian_in_leaf": [0.01, 0.05, 0.1],
                            "min_data_in_leaf": [60, 120, 240],
                            "eta": [0.01, 0.05, 0.1, 0.5],
                            "learning_rate": [0.01, 0.05, 0.1, 0.5],
                            "lambda_l1": [0.5, 0.7, 0.9, 1.3],
                            "lambda_l2": [0.5, 0.7, 0.9, 1.3],
                            "reg_alpha": [0.5, 0.7, 0.9, 1.3],
                            "reg_lambda": [0.5, 0.7, 0.9, 1.3],
                            "path_smooth": [0.5, 0.7, 0.9, 1.3],
                            # "bagging_fraction": [0.5, 0.7, 0.8],
                            # "feature_fraction": [0.5, 0.7, 0.8],
                            "colsample_bytree": [0.5, 0.7],
                        }
                        ),
    )
        
    def __init__(self, target_var, predictives, suffix="", logger=Logger()):
        self.logger = logger
        self.suffix = suffix
        self.axis_limit = [1e10, 0]
        self.models = {}
        # self.tokenizer = DistilBertTokenizer.from_pretrained(self.MODELLING_CONFIG["PRE_TRAINED_MODEL_NAME"])
        # self.bert_model = DistilBertModel.from_pretrained(self.MODELLING_CONFIG["PRE_TRAINED_MODEL_NAME"])
        
        self.meta = dict(
            predictives = predictives,
            target_var = target_var,
            suffix = suffix,
            stime = datetime.now(),
            user = getpass.getuser(),
            sys = uname()[1],
            py = ".".join(map(str, sys.version_info[:3])),
        )
        
    
    def prepare_model_data(self):
        self.logger.info("Using Reviews Data For Model Training on Sentiment Analysis:")
        
        self.logger.info("  reading reviews data...")
        fname = os.path.join(self.FILES["PREPROCESS_DIR"], self.FILES["REVIEWS_ABT_FILE"])
        self.data["reviews_abt"] = pd.read_parquet("{}.parquet".format(fname))
        
        self.logger.info("  drop those rows with non-missing rating...")
        self.data["reviews_abt"] = self.data["reviews_abt"][self.data["reviews_abt"]["rating"].notnull()]
        
        self.logger.info("  encode sentiments to rating...")
        self.data["reviews_abt"][self.meta["target_var"]] = self.data["reviews_abt"]["rating_encode"].apply(lambda x: self.sentiment_encoder(x))
        
        
    def run(self, data, algorithms=["LGBMC"]):
        if data is None:
            data = self.data["reviews_abt"]
            
        self.classification(data, algorithms)
        self.sort_models()
        self.get_results()
        self.save_models()
        self.meta["runtime"] = datetime.now() - self.meta["stime"]
        self.meta["algorithms"] = algorithms
        self.logger.info("Complete training in {}.".format(self.meta["runtime"]))
        
        
    def classification(self, data, algorithms):
        self.logger.info("Extract Features from {} Data:".format(self.meta["predictives"]))
        self.logger.info("  create text features from TF-IDF vectorizer...")
        self.tfidf_vector = TfidfVectorizer(tokenizer=self.preprocess)

        self.logger.info("  fragment 'reviews' & 'sentiment' data into 'X' & 'y' for training...")
        X = self.tfidf_vector.fit_transform(data[self.meta["predictives"]])
        y = data[self.meta["target_var"]]
        X_train, X_test, X_val, y_train, y_test, y_val = self.random_split(X, y)
        
        for i, algorithm in enumerate(algorithms):
            start = time.time()
            self.logger.info("  >---------------------------------------------------------------------------------------------<")
            self.logger.info("  {}. trained using algorithm '{}' with data shape of {}".format(i, algorithm, data.shape))
            self.logger.info("  >---------------------------------------------------------------------------------------------<")
            
            if not algorithm in self.models:
                self.models[algorithm] = []
            
            model = Model(self.CLASSFICATION_ALGORITHMS[algorithm], self.meta["target_var"], self.meta["predictives"])
            model.set_props(algorithm, data)
            model.classification_tree(X_train, X_test, X_val, y_train, y_test, y_val)
            self.models[algorithm]= model.get_meta()
            
            self.logger.info("      Train Metrics:: {}".format(", ".join(["{}:{}".format(m, v) for m, v in 
                                                                          self.models[algorithm]["train_metrics"].items() 
                                                                          if (m == "ACCURACY") or (m == "PRECISION") or 
                                                                          (m == "RECALL") or (m == "F1_SCORE")])))
            self.logger.info("      Test Metrics:: {}".format(", ".join(["{}:{}".format(m, v) for m, v in 
                                                                         self.models[algorithm]["test_metrics"].items() 
                                                                         if (m == "ACCURACY") or (m == "PRECISION") or 
                                                                         (m == "RECALL") or (m == "F1_SCORE")])))
            
        self.logger.info("    Complete training using in {:.2f}s".format(time.time()-start))
            
        
    def sentiment_encoder(self, rating):
        if rating == "positive":
            return 0
        elif rating == "neutral":
            return 1
        elif rating == "negative":
            return 2
        else: 
            return np.nan
        
    
    def set_props(self, alg, df):
        self.algorithm = alg
        self.n_records  = df.shape[0]
        
        
    def sentence_tokenize(self, text):
        return nltk.sent_tokenize(text)


    def word_tokenize(self, text):
        return nltk.word_tokenize(text)


    def preprocess(self, text):
        sentence_tokens = self.sentence_tokenize(text)
        word_list = []
        for each_sent in sentence_tokens:
            word_tokens = self.word_tokenize(each_sent)
            for i in word_tokens:
                    word_list.append(i)
                    
        return word_list
    
    
    def random_split(self, X, y):
        self.logger.info("Splitting Train & Test Data:")
        self.logger.info("  train-test split at ratio: {}...".format(self.MODELLING_CONFIG["TEST_SPLIT_RATIO"]))
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=self.MODELLING_CONFIG["TEST_SPLIT_RATIO"], 
                                                            shuffle=True,
                                                            stratify=y,
                                                            random_state=self.MODELLING_CONFIG["RANDOM_SEED"])

        self.logger.info("  train-validation split at ratio: {}...".format(self.MODELLING_CONFIG["VALIDATE_SPLIT_RATIO"]))
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                        test_size=self.MODELLING_CONFIG["VALIDATE_SPLIT_RATIO"], 
                                                        shuffle=True,
                                                        stratify=y_train,
                                                        random_state=self.MODELLING_CONFIG["RANDOM_SEED"])
        
        return X_train, X_test, X_val, y_train, y_test, y_val
        
        
    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        ds = ReviewDataset(reviews=df[self.meta["predictives"]].to_numpy(), targets=df[self.meta["target_var"]].to_numpy(), tokenizer=tokenizer, max_len=max_len)

        return DataLoader(ds, batch_size=batch_size, num_workers=4)
    
    
    # # Visualization
    @staticmethod
    def histogram_plot(xvar, xlabel, title=None):
        fig, ax = plt.subplots(figsize=(15,4))
        sns.histplot(x=xvar, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=12, weight="bold")
        
        return fig
    
    
    @staticmethod
    def confusion_matrix_plot(cf_matrix):
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt=".2%", cmap="binary", ax=ax)
        ax.set_title("Confusion Matrix from Test Data Evaluation", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        
        return fig
    
    
    def sort_models(self):
        self.meta["METRIC_BEST"] = self.MODELLING_CONFIG["METRIC_BEST"]
        self.meta["METRIC_BEST_THRESHOLD"] = self.MODELLING_CONFIG.get("METRIC_BEST_THRESHOLD", None)

        self.logger.info("  sorting models base on metric '{}'".format(self.meta["METRIC_BEST"]))
        reverse = False if self.meta["METRIC_BEST"] in ["MAE", "MAPE", "RMSE", "MSE"] else True

        self.best_algorithms = []
        self.models = dict(sorted(self.models.items(), key=lambda x: x[1]["test_metrics"][self.meta["METRIC_BEST"]].mean()))

        for algo, comp in self.models.items():
            metric_value = round(comp["test_metrics"][self.meta["METRIC_BEST"]].mean(), 2)
            if (not reverse and metric_value < self.meta["METRIC_BEST_THRESHOLD"] ) or \
            (reverse and metric_value > self.meta["METRIC_BEST_THRESHOLD"] ):
                self.best_algorithms.append(algo)
                
            # min_x = min(self.models[algo]["y_test"].min(), self.models[algo]["test_pred"].min())
            # if min_x < self.axis_limit[0]:
            #     self.axis_limit[0] = min_x
            # max_x = max(self.models[algo]["y_test"].max(), self.models[algo]["test_pred"].max())            
            # if max_x > self.axis_limit[1]:
            #     self.axis_limit[1] = max_x
            
            
    def get_results(self):
        metrics_dict = {}
        self.data["metrics_df"] = pd.DataFrame()

        for algo in self.models:
            temp_df = pd.DataFrame.from_dict(self.models[algo]["test_metrics"], orient="index").T
            temp_df["MODEL"] = algo
            self.data["metrics_df"] = pd.concat([self.data["metrics_df"], temp_df])
    
    
    def save_models(self, fname=""):
        self.meta["n_models"] = len(self.models)
        training = dict(
            models = {m: [self.models[m]["alg"]] for m, v in self.models.items()},
            meta = self.meta,
        )

        model_id = str(datetime.now()).split(" ")[0].replace("-", "_") + \
            str(datetime.now()).split(" ")[1].split(".")[0].replace(":", "_")

        if fname == "":
            suffix = self.meta["suffix"]
            if self.suffix != "":
                suffix += "_" + self.suffix
            fname = os.path.join(self.FILES["MODEL_DATA_DIR"], self.meta["target_var"] 
                                 + "_" + suffix + "_" + model_id + ".pickle")

        if os.path.exists(fname):
            os.remove(fname)
        with open(fname, "wb") as handle:
            pickle.dump(training, handle, protocol=pickle.HIGHEST_PROTOCOL)            
            self.logger.info("  training and its models saved to file '{}'.".format(fname))
    
    
class ReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    
    def __len__(self):
        return len(self.reviews)
    
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(review,
                                              add_special_tokens=True,
                                              max_length=self.max_len,
                                              return_token_type_ids=False,
                                              pad_to_max_length=True,
                                              return_attention_mask=True,
                                              return_tensors="pt",
                                              )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long)
            }