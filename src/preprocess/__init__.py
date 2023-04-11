import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import trange
from typing import Any, List, Dict
from collections import OrderedDict, Counter
from wordcloud import ImageColorGenerator, WordCloud, STOPWORDS
from IPython.display import display, Markdown, HTML, clear_output, display_html

import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt

from unidecode import unidecode
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.config import Config

class Logger():
    info = print
    warnings = print
    critical = print
    error = print
    
    
class Analysis(Config):
    data = {}
    
    def __init__(self, logger=Logger()):
        self.logger = logger
        
        
    @staticmethod
    def vars(types:List=[], wc_vars:str=[], qreturn_dict:Any=False):
        if types==None:
            types = [V for V in Config.VARS]
        selected_vars = []
        for t in types:
            for d in Config.VARS[t]:
                if not d.get("predictive"):
                    continue
                if len(wc_vars) != 0: 
                    matched_vars = fnmatch.filter(wc_vars, d["var"])
                    if qreturn_dict:
                        for v in matched_vars:
                            dd = d.copy()
                            dd["var"] = v 
                            if not dd in selected_vars:
                                selected_vars.append(dd)
                    else:
                        for v in matched_vars:
                            if not v in selected_vars:
                                selected_vars.append(v)
                else:
                    if qreturn_dict and not d in selected_vars:
                        selected_vars.append(d)
                    else:
                        if not d["var"] in selected_vars:
                            selected_vars.append(d["var"])
        return selected_vars
    
    
    def get_reviews_data(self):
        start = time.time()
        self._get_readings_reviews(read_preprocess=self.ANALYSIS_CONFIG["READ_PREPROCESS"])
        
        self._get_goodreads_reviews(read_preprocess=self.ANALYSIS_CONFIG["READ_PREPROCESS"])
        
        self.logger.info("  mnerge both Readings & GoodReads reviews data...")
        self.data["reviews_abt"] = pd.concat([self.data["readings_reviews"], self.data["goodreads_reviews"]])
        self.data["reviews_abt"] = self.data["reviews_abt"].reset_index(drop=True)
        self.data["reviews_abt"]["rating"] = self.data["reviews_abt"]["rating"].astype(float)
        
        self.logger.info("  encode rating from Goodreads into 3 level of sentiments, 'negative', 'neutral', 'positive'...")
        self.data["reviews_abt"]["rating_encode"] = self.data["reviews_abt"]["rating"].apply(lambda x: self.rating_encode(x))
        
        for col in self.data["reviews_abt"].columns:
            if col in self.vars(types=["REVIEWS"]):
                col_dtypes = [sub_dict for sub_dict in self.vars(types=["REVIEWS"], qreturn_dict=True) if sub_dict["var"]==col][0].get("dtypes")
                self.data["reviews_abt"][col] = self.data["reviews_abt"][col].astype(col_dtypes)

        if self.QDEBUG:
            fname = os.path.join(self.FILES["PREPROCESS_DIR"], self.FILES["REVIEWS_ABT_FILE"])
            self.data["reviews_abt"].to_parquet("{}.parquet".format(fname))
        
        self.logger.info("  done creating reviews ABT data using {:.2f}s...".format(time.time() - start))
        
    
    def _get_readings_reviews(self, read_preprocess:Any=False):
        start = time.time()
        
        if read_preprocess:
            fname = os.path.join(self.FILES["PREPROCESS_DIR"], "process_readings_reviews")
            self.data["readings_reviews"] = pd.read_parquet("{}.parquet".format(fname))
        else:
            self.logger.info("Reading Readings.au reviews data from {} directory:".format(self.FILES["RAW_DATA_DIR"]))
            readings_files = [file for file in os.listdir(self.FILES["RAW_DATA_DIR"]) if ("raw_readings" in file) and ("parquet" in file)]
            
            self.data["readings_reviews"] = pd.DataFrame()
            for file in readings_files:
                self.logger.info("    > {}".format(file))
                temp_file = pd.read_parquet(os.path.join(self.FILES["RAW_DATA_DIR"], file))
                self.data["readings_reviews"] = pd.concat([self.data["readings_reviews"], temp_file])
                
            self.data["readings_reviews"] = self.data["readings_reviews"].dropna(subset=["reviews"]).reset_index(drop=True)
            self.data["readings_reviews"]["title"] = self.data["readings_reviews"]["book_author"].apply(lambda x: x.split("by")[0].strip())

            self.logger.info("  acquire the reviews sentence length...")
            self.data["readings_reviews"]["reviews_length"] = self.data["readings_reviews"]["reviews"].apply(lambda x: len(x))

            self.logger.info("  preprocess the reviews test...")
            self.logger.info("    > lower the text. remove white space, symbols & other special symbols")
            self.logger.info("    > remove stop words")
            self.logger.info("    > apply Krovertz stemming from tokenized words")
            self.logger.info("    > apply lemmatization from tokenized words")
            self.data["readings_reviews"]["reviews"] = self.data["readings_reviews"]["reviews"].apply(lambda x: self._preprocess_reviews_text(x))

            if self.QDEBUG:
                fname = os.path.join(self.FILES["PREPROCESS_DIR"], self.FILES["PREPROCESS_READINGS_FILE"])
                self.data["readings_reviews"].to_parquet("{}.parquet".format(fname))
        
        self.logger.info("  done loading & preprocess READINGS.au reviews data using {:.2f}s...".format(time.time() - start))
            
    
    def _get_goodreads_reviews(self, read_preprocess:Any=False):
        start = time.time()
        
        if read_preprocess:
            fname = os.path.join(self.FILES["PREPROCESS_DIR"], "process_goodreads_reviews")
            self.data["goodreads_reviews"] = pd.read_parquet("{}.parquet".format(fname))
        else:
            self.logger.info("Reading raw Goodreads reviews data from {} directory:".format(self.FILES["RAW_DATA_DIR"]))
            goodreads_files = [file for file in os.listdir(self.FILES["RAW_DATA_DIR"]) if ("raw_goodreads" in file) and ("parquet" in file)]
            
            self.data["goodreads_reviews"] = pd.DataFrame()
            for file in goodreads_files:
                self.logger.info("    > {}".format(file))
                temp_file = pd.read_parquet(os.path.join(self.FILES["RAW_DATA_DIR"], file))
                self.data["goodreads_reviews"] = pd.concat([self.data["goodreads_reviews"], temp_file])
            
            self.data["goodreads_reviews"] = self.data["goodreads_reviews"].dropna(subset=["reviews"]).reset_index(drop=True)
            self.logger.info("  acquire the reviews sentence length...")
            self.data["goodreads_reviews"]["reviews_length"] = self.data["goodreads_reviews"]["reviews"].apply(lambda x: len(x))

            self.logger.info("  preprocess the reviews test...")
            self.logger.info("    > lower the text. remove white space, symbols & other special symbols")
            self.logger.info("    > remove stop words")
            self.logger.info("    > apply Krovertz stemming from tokenized words")
            self.logger.info("    > apply lemmatization from tokenized words")
            self.data["goodreads_reviews"]["reviews"] = self.data["goodreads_reviews"]["reviews"].apply(lambda x: self._preprocess_reviews_text(x))

            if self.QDEBUG:
                fname = os.path.join(self.FILES["PREPROCESS_DIR"], self.FILES["PREPROCESS_GOODREADS_FILE"])
                self.data["goodreads_reviews"].to_parquet("{}.parquet".format(fname))
            
        self.logger.info("  done loading & preprocess GOODREADS reviews data using {:.2f}s...".format(time.time() - start))
        
        
    def _preprocess_reviews_text(self, reviews):
        reviews = re.sub(r"\d+", "", str(reviews).lower().strip())
        reviews = re.sub(r"[\W\s]", " ", reviews)
        reviews = re.sub(r"\<a href", " ", reviews)
        reviews = re.sub(r"&amp;", " ", reviews)
        reviews = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', " ", reviews)
        reviews = re.sub(r"<br />", " ", reviews)
        reviews = re.sub(r"\'", " ", reviews)
        
        reviews = remove_stopwords(" ".join([word for word in reviews.split() if word not in self.ANALYSIS_CONFIG["NLTK_STOPWORDS"]]))
        reviews = " ".join([word for word in reviews.split() if word not in ENGLISH_STOP_WORDS])
        
        reviews = " ".join([self.ANALYSIS_CONFIG["KROVERTZ_STEMMER"].stem(unidecode(word)) for word in word_tokenize(reviews)])

        reviews = " ".join([self.ANALYSIS_CONFIG["LEMMATIZER"].lemmatize(word) for word in word_tokenize(reviews)])
        
        return reviews
    
    
    # # Preprocessing
    def rating_encode(self, rating):
        if rating < 3:
            return "negative"
        elif 3 <= rating < 4:
            return "neutral"
        elif rating >= 4:
            return "positive"
        else:
            return np.nan
        
        
    def corpus_list(self, text):
        text_list = text.split()
        
        return text_list
    
    
    def term_frequency_analysis(self, df, reviews_col, top_sample_size=20):
        df["review_lists"] = df[reviews_col].apply(self.corpus_list)

        corpus = []
        for i in trange(df.shape[0], ncols=150, nrows=10, colour="green", smoothing=0.8):
            corpus += df["review_lists"][i]
        most_common = Counter(corpus).most_common(20)

        words = []
        freq = []
        for word, count in most_common:
            words.append(word)
            freq.append(count)

        most_common_df = pd.DataFrame(data=zip(words, freq), columns=["words", "frequency"])
        
        return most_common_df
    
    
    def create_n_grams(self, df, reviews_col, ngram_range):
        cv = CountVectorizer(ngram_range=ngram_range)
        ngrams = cv.fit_transform(df[reviews_col])

        count_values = ngrams.toarray().sum(axis=0)
        ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv.vocabulary_.items()], reverse=True))
        ngram_freq.columns = ["frequency", "ngram"]
        
        return ngram_freq
        
    
    # # Visualization  
    def vertical_bar_plot(self, df, xvar, yvar, title=None):
        fig, ax = plt.subplots(figsize=(7,4))
        sns.barplot(data=df, x=xvar, y=yvar, ax=ax)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel(xvar, fontsize=10)
        ax.set_ylabel(yvar, fontsize=10)
        
        return fig
    
    
    def distribution_plot(self, df, xvar, title=None):
        fig, ax = plt.subplots(figsize=(10,4))
        sns.histplot(data=df, x=xvar, bins=50, ax=ax)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel(xvar, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        
        return fig
    
    
    def horizontal_bar_plot(self, df, xvar, yvar, title=None):
        fig, ax = plt.subplots(figsize=(10,10))
        sns.barplot(data=df, x=xvar, y=yvar, ax=ax)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel(xvar, fontsize=10)
        ax.set_ylabel(yvar, fontsize=10)
        
        return fig
    
    
    def wordcloud_plot(self, text):
        stopwords = set(STOPWORDS)

        wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400).generate(text)
        plt.figure(figsize=(40,20))
        plt.tight_layout(pad=0)
        plt.imshow(wordcloud, interpolation="bilinear")
        
        return plt.gcf()
    
    
    def grid_df_display(self, list_dfs, rows=1, cols=2, fill:str="cols"):
        html_table = "<table style = 'width: 100%; border: 0px'> {content} </table>"
        html_row = "<tr style = 'border:0px'> {content} </tr>"
        html_cell = "<td style='width: {width}%; vertical-align: top; border: 0px'> {{content}} </td>"
        html_cell = html_cell.format(width=8000)

        cells = [ html_cell.format(content=df.to_html()) for df in list_dfs[:rows*cols] ]
        cells += cols * [html_cell.format(content="")]

        if fill == 'rows':
            grid = [ html_row.format(content="".join(cells[i: i+cols])) for i in range(0,rows*cols, cols)]

        if fill == 'cols': 
            grid = [ html_row.format(content="".join(cells[i: rows*cols:rows])) for i in range(0,rows)]
            
        dfs = display(HTML(html_table.format(content="".join(grid))))
        
        return dfs
    
    
    def descriptive_data(self, df:pd.DataFrame) -> pd.DataFrame:
        descriptive_info = {"Number of Books: ": df["title"].nunique(),
                            "No. of Variables: ": int(len(df.columns)),
                            "No. of Observations: ": int(df.shape[0]),
                            }
        descriptive_df = pd.DataFrame(descriptive_info.items(), columns=["Descriptions", "Values"]).set_index("Descriptions")
        descriptive_df.columns.names = ["Data Statistics"]
        
        return descriptive_df

    
    def data_type_analysis(self, df:pd.DataFrame) -> pd.DataFrame:
        categorical_df = pd.DataFrame(df.reset_index(inplace=False).dtypes.value_counts())
        categorical_df.reset_index(inplace=True)

        categorical_df = categorical_df.rename(columns={"index": "Types", 0:"Values"})
        categorical_df["Types"] = categorical_df["Types"].astype(str)
        categorical_df = categorical_df.set_index("Types")
        categorical_df.columns.names = ["Variables"]

        return categorical_df