import os
import inspect
import fnmatch
import evaluate
from collections import OrderedDict

from nltk.corpus import stopwords
from krovetzstemmer import Stemmer
from nltk.stem import WordNetLemmatizer

base_path, current_dir = os.path.split(os.path.dirname(inspect.getfile(inspect.currentframe())))

class Config(object):
    QDEBUG = True
    
    FILES = dict(
        DATA_LOCAL_DIR  = os.path.join(base_path, "data"),
        RAW_DATA_DIR    = os.path.join(base_path, "data", "raw"),
        PREPROCESS_DIR  = os.path.join(base_path, "data", "preprocess"),
        MODEL_DATA_DIR  = os.path.join(base_path, "data", "models"),
        
        RAW_GOODREADS_FILE          = "raw_goodreads_reviews",
        RAW_READINGS_FILE           = "raw_readings_reviews",
        PREPROCESS_GOODREADS_FILE   = "process_goodreads_reviews",
        PREPROCESS_READINGS_FILE    = "process_readings_reviews",
        REVIEWS_ABT_FILE            = "reviews_abt",
        REVIEWS_ABT_TRAIN           = "reviews_abt_train",
        REVIEWS_ABT_TEST            = "reviews_abt_test",
        
        X_TRAIN = "x_train_reviews",
        X_TEST  = "x_test_reviews",
        X_VAL   = "x_val_reviews",
        Y_TRAIN = "y_train_sentiments",
        Y_TEST  = "y_test_sentiments",
        Y_VAL   = "y_val_sentiments",
        
    )
    
    CRAWLER_CONFIG = dict(
        URL         = "https://www.goodreads.com/book/show", # "https://www.readings.com.au/reviews"
        LOWER_LIMIT = 18000, # 1
        UPPER_LIMIT = 21000, # goodreads=4532, readings max=550
    )
    
    ANALYSIS_CONFIG = dict(
        READ_PREPROCESS     = True,
        NLTK_STOPWORDS      = stopwords.words("english"),
        KROVERTZ_STEMMER    = Stemmer(),
        LEMMATIZER          = WordNetLemmatizer(),
    )
    
    
    MODELLING_CONFIG = dict(
        RANDOM_SEED             = 42,
        EARLY_STOPPING_ROUND    = 5,
        TEST_SPLIT_RATIO        = 0.2,
        VALIDATE_SPLIT_RATIO    = 0.2,
        TUNING_CV_METRICS       = "f1",
        TUNING_CV_SPLIT         = 5,
        METRIC_BEST             = "F1_SCORE",
        METRIC_BEST_THRESHOLD   = 0.4,
        
        PRE_TRAINED_MODEL_NAME  = "bert-base-uncased", # distilbert-base-uncased, bert-base-cased
        MAX_LEN                 = 256,
        MAX_LEN_TRUNCATION      = True,
        BATCH_SIZE              = 8,
        CLASS_NAMES             = ["negative", "neutral", "positive"],
    )
    
    
    VARS = OrderedDict(
        REVIEWS = [
            dict(var="rating",          dtypes=float,   predictive=False),
            dict(var="page_number",     dtypes=int,     predictive=True),
            dict(var="book_author",     dtypes=str,     predictive=True),
            dict(var="reviews",         dtypes=str,     predictive=True),
            dict(var="title",           dtypes=str,     predictive=True),
            dict(var="reviews_length",  dtypes=int,     predictive=True),
        ],
        
    )