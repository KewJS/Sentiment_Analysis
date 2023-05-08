import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import nltk

from sklearn.utils import class_weight
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import Config


class Logger():
    info = print
    warnings = print
    critical = print
    error = print


class Model(Config):
    def __init__(self, m, target_var, predictives, logger=Logger()):
        self.target_var = target_var   # the reponse variable
        self.predictives = predictives # the predictive variables
        if target_var in self.predictives:
            self.predictives.remove(target_var)
        self.logger = logger
        
        self.date = None
        self.alg = m["alg"](**m["args"])  # clone the base model (remove fitted data)
        self.param_grid = m["param_grid"] if "param_grid" in m else None
        self.param_opt = m["param_opt"] if "param_opt" in m else None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.train_pred = None
        self.test_pred = None
        self.train_metrics = {}
        self.test_metrics = {}
        self.created = datetime.now()
        self.modified = datetime.now()
        self.model_data = None
        
        
    def get_meta(self):
        return dict(
            alg = self.alg,
            predictives = self.predictives,
            n_records = self.n_records,
            class_weights = self.sentiments_weight_dict,
            train_error_sd = self.train_error_sd,
            test_error_sd = self.test_error_sd,
            train_metrics = self.train_metrics,
            test_metrics = self.test_metrics,
            y_train = self.y_train,
            train_pred = self.train_pred,
            y_test = self.y_test,
            test_pred = self.test_pred,
            y_val = self.y_val,
            created = self.created,
            modified = self.modified,
        )
        
        
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
    
    
    def evaluate(self, y_test, y_pred):
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred,)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        metrics = dict(ACCURACY=acc, PRECISION=prec, RECALL=recall, F1_SCORE=f1, CM=cm)

        return metrics


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


    def classification_tree(self, X_train, X_test, X_val, y_train, y_test, y_val):
        # self.logger.info("  create text features from TF-IDF vectorizer...")
        # self.tfidf_vector = TfidfVectorizer(tokenizer=self.preprocess)

        # self.logger.info("  extract the 'reviews' & 'sentiment' data for training...")
        # X = self.tfidf_vector.fit_transform(data[self.predictives])
        # y = data[self.target_var]
        # X_train, X_test, X_val, y_train, y_test, y_val = self.random_split(X, y)
        self.n_records = X_train.shape
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        
        self.logger.info("  define class weights...")
        self.sentiments_weight = class_weight.compute_class_weight(class_weight="balanced", 
                                                                   classes=np.unique(y_train),
                                                                   y=y_train
                                                                   )
        self.sentiments_weight_dict = dict(zip(np.unique(y_train), self.sentiments_weight))
        
        if self.param_grid != None:
            self.logger.info("  running hyperparameter tuning...")
            n_folds = StratifiedKFold(n_splits=self.MODELLING_CONFIG["TUNING_CV_SPLIT"], shuffle=True, random_state=42)
            random_search = RandomizedSearchCV(estimator=self.alg, param_distributions=self.param_grid,
                                                scoring="f1", cv=n_folds, n_jobs=-1)
            random_search.fit(X_train, y_train, eval_set=([(X_train, y_train), (X_val, y_val)]))
            if hasattr(self.alg, "sample_weight"):
                random_search.best_params_["sample_weight"] = self.sentiments_weight
            elif hasattr(self.alg, "class_weight"):
                random_search.best_params_["class_weight"] = self.sentiments_weight_dict
            self.alg = self.alg.set_params(**random_search.best_params_)
        
        self.alg.fit(X_train, y_train.values.ravel(), eval_set=([(X_train, y_train), (X_val, y_val)]))
        self.train_pred = self.alg.predict(X_train)
        self.train_error_sd = np.std(abs(self.train_pred - y_train))
        self.train_metrics = self.evaluate(y_train, self.train_pred)
        
        self.test_pred = self.alg.predict(X_test)
        self.test_error_sd = np.std(abs(self.test_pred - y_test))
        self.test_metrics = self.evaluate(y_test, self.test_pred)
        
        
    def feature_importance_plot(self):
        fig, ax = plt.subplots(figsize=(10, len(self.predictives)/2))
        #ax.bar(self.predictives, self.alg.feature_importances_)
        #ax.set_xticklabels(self.predictives, rotation = 45)

        s = pd.Series(self.alg.feature_importances_, index=self.predictives)
        ax = s.sort_values(ascending=False).plot.barh()
        ax.invert_yaxis()

        patches = [mpatches.Patch(label="Test Size: {}".format(self.actual.shape[0]), color='none')]
        for alg, val in self.metrics.items():
            patches.append(mpatches.Patch(label="{}: {:0.2f}".format(alg, val), color='none',))
        plt.legend(handles=patches, loc='lower right')
 
        return fig


    def residual_plot(self):
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[3, 1])

        ax1 = fig.add_subplot(gs[0])
        residual = self.actual - self.pred
        sns.residplot(x=self.pred, y=residual, ax=ax1)
        # ax.scatter(self.pred, residual)
        ax1.set_ylabel("Residual")
        ax1.set_xlabel("Predict")
        ax1.set_title(self.name)

        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        ax2.hist(residual, orientation="horizontal")
        ax2.set_xlabel('Residual Distribution')
 
        return fig 