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
        
        TRAIN_ABT                   = "train_reviews",
        TEST_ABT                    = "test_reviews",
        VAL_ABT                     = "val_reviews",
        
    )
    
    CRAWLER_CONFIG = dict(
        URL         = "https://www.goodreads.com/book/show", # "https://www.readings.com.au/reviews"
        LOWER_LIMIT = 14500, # 1
        UPPER_LIMIT = 16000, # goodreads=4532, readings max=550
    )
    
    ANALYSIS_CONFIG = dict(
        READ_PREPROCESS     = True,
        NLTK_STOPWORDS      = stopwords.words("english"),
        KROVERTZ_STEMMER    = Stemmer(),
        LEMMATIZER          = WordNetLemmatizer(),
    )
    
    
    MODELLING_CONFIG = dict(
        RANDOM_SEED             = 42,
        TEST_SPLIT_RATIO        = 0.2,
        VALIDATE_SPLIT_RATIO    = 0.5,
        PRE_TRAINED_MODEL_NAME  = "bert-base-cased",
        MAX_LEN                 = 512,
        MAX_LEN_TRUNCATION      = True,
        BATCH_SIZE              = 16,
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