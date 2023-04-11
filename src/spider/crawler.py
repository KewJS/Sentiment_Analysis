import re
import os
import time
import pickle
import requests
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup as bs

from src.config import Config

class Logger(object):
    info = print
    warning = print
    critical = print
    error = print
    

class Crawler(Config):
    data = {}
    
    def __init__(self, url="", suffix="", logger=Logger()):
        if url == "":
            self.url = "https://www.goodreads.com/book/show"
        else:
            self.url = url # "https://www.goodreads.com/book/show", "https://www.readings.com.au/reviews"
        self.suffix = suffix
        self.logger = logger
        self.low_limit = self.CRAWLER_CONFIG["LOWER_LIMIT"] 
        self.high_limit = self.CRAWLER_CONFIG["UPPER_LIMIT"]
        
        
    def web_spiders(self, domain):
        start = time.time()
        self.data["reviews_df"] = pd.DataFrame()
        
        self.logger.info("Initiating Reviews Scraping from Web Page: {}".format(self.url))
        for i in range(self.low_limit, self.high_limit):
            
            if domain == "goodreads":
                web_url = "{}/{}".format(self.url, i)
            elif domain == "readings":
                if i == 1:
                    web_url = self.url
                else:
                    web_url = "{}?page={}".format(self.url, i)
            
            page = requests.get(web_url)
            soup = bs(page.content, "html.parser")
            
            if domain == "goodreads":
                self.sub_reviews_dict = {}
                self.reviews_content_dict = {}
                self.data["sub_reviews_df"] = pd.DataFrame()
                
                title_elements = soup.findAll("h1", {"class" : re.compile("Text Text__title1")})
                if len(title_elements) == 0:
                    self.logger.info("   {} at page {} has no data, skipping...".format(web_url, i))
                    continue
                else:
                    self.logger.info("  extracting data from {} at page {}...".format(web_url, i))
                    self.sub_reviews_dict["title"] = title_elements[0].text

                    rating_elements = soup.findAll("div", {"class" : re.compile("RatingStatistics__column")})
                    self.sub_reviews_dict["rating"] = rating_elements[0].text

                    author_elements = soup.findAll("span", {"class" : re.compile("ContributorLink__name")})
                    self.sub_reviews_dict["book_author"] = author_elements[0].text

                    self.sub_reviews_dict["page_number"] = i

                    reviews_elements = soup.findAll("section", {"class" : re.compile("ReviewText")})
                    for i, content in enumerate(reviews_elements):
                        self.reviews_content_dict[i] = content.text
                        
                    reviews_content_df = pd.DataFrame.from_dict(self.reviews_content_dict, orient="index").drop_duplicates().reset_index(drop=True)
                    # self.data["sub_reviews_df"] = self.data["sub_reviews_df"].append(self.sub_reviews_dict, ignore_index=True)
                    self.data["sub_reviews_df"] = pd.concat([self.data["sub_reviews_df"], reviews_content_df], axis=1).ffill()
                    self.data["reviews_df"] = pd.concat([self.data["reviews_df"], self.data["sub_reviews_df"]])
                        
            elif domain == "readings":
                reviews_elements = soup.findAll("div", {"class" : re.compile("review-index-summary review-index-summary--odd review-index-summary--*")})
                
                self.logger.info("  extracting data from {} at page {}...".format(web_url, i))
                for content in reviews_elements:
                    self.sub_reviews_dict = {}
                    self.data["sub_reviews_df"] = pd.DataFrame()

                    reviews_main = content.find_all("div", {"class": "review-index-summary__content copy"})
                    review_links = [item.find("a").get("href") for item in reviews_main]
                    temp_book = review_links[0].split("/")[2]
                    temp_link = "{}/{}".format(self.url, temp_book)

                    review_page = requests.get(temp_link)
                    soup = bs(review_page.content, "html.parser")

                    reviews_content = soup.findAll("div", {"class" : re.compile("review-post__content copy")})
                    content_elements = [content.find_all("p") for content in reviews_content]
                    try:
                        book_title_elements = soup.select("h1.review-post__title")[0].text
                        temp_reviews = " ".join([p.get_text() for p in content_elements[0]])
                        temp_book_author = review_links[0].split("/")[2].replace("-", " ")
                    except:
                        self.logger.info("  > there are issue in extracting data from page {}...".format(i))
                        book_title_elements = np.nan
                        temp_reviews = np.nan
                        temp_book_author = np.nan

                    self.sub_reviews_dict["page_number"] = i
                    self.sub_reviews_dict["book_author"] = book_title_elements
                    self.sub_reviews_dict["reviews"] = temp_reviews
                    
                    reviews_content_df = pd.DataFrame.from_dict(self.reviews_content_dict, orient="index").drop_duplicates().reset_index(drop=True)
                    # self.data["sub_reviews_df"] = self.data["sub_reviews_df"].append(self.sub_reviews_dict, ignore_index=True)
                    self.data["sub_reviews_df"] = pd.concat([self.data["sub_reviews_df"], reviews_content_df], axis=1).ffill()
                    # self.data["sub_reviews_df"] = self.data["sub_reviews_df"].reset_index(drop=True)
                    self.data["reviews_df"] = pd.concat([self.data["reviews_df"], self.data["sub_reviews_df"]])
        
        self.data["reviews_df"] = self.data["reviews_df"].reset_index(drop=True)
        if 0 in self.data["reviews_df"].columns:
            self.data["reviews_df"] = self.data["reviews_df"].rename(columns={0:"reviews"})
        self.logger.info("  complete scraping raw data using total time of {:.2f}s...".format(time.time() - start))
        
        if self.QDEBUG:
            fname = os.path.join(self.FILES["RAW_DATA_DIR"], "raw_{}_{}_{}_reviews".format(domain, self.CRAWLER_CONFIG["LOWER_LIMIT"], self.CRAWLER_CONFIG["UPPER_LIMIT"]))
            self.data["reviews_df"].to_csv("{}{}.csv".format(fname, self.suffix))
            self.data["reviews_df"].to_parquet("{}.parquet".format(fname))
            
            
    def download_cnn_data(self):
        self.logger.info("Downloading Raw CNN Data from TFDS:")
        cnn_df = tfds.as_numpy(tfds.load(
            "cnn_dailymail",
            split="test",
            batch_size=-1
        ))
        
        return cnn_df
    
    
    def create_raw_cnn_data(cnn_data):
        self.logger.info("Preprocessing Raw CNN Data:")
        cnn_df = pd.DataFrame(cnn_data)
        cnn_df.highlights = cnn_df.highlights.apply(lambda x: x.decode('utf-8')) 
        cnn_df["summary"] = cnn_df.highlights.apply(lambda x: "".join(x.split("\n")) ) 
        cnn_df.article = cnn_df.article.apply(lambda x: x.decode('utf-8')) 
        cnn_df["art_sents"] = cnn_df.article.apply(lambda x: len([x for x in nlp(x).sents])) 
        
        logger.info("Savings CNN data to {} directory:".format(RAW_DATA_PATH))
        cnn_df.to_parquet(os.path.join(RAW_DATA_PATH, "cnn_articles.parquet"))
        
        return cnn_df

        
    def goodreads_web_crawler(self, web_url="https://www.goodreads.com/book/show", low_range=1, high_range=4532):
        start = time.time()
        reviews_df = pd.DataFrame()

        self.logger.info("Initiating Reviews Scraping from web page:")
        for i in range(low_range, high_range): # 18100000
            sub_reviews_dict = {}
            self.reviews_content_dict = {}
            sub_reviews_df = pd.DataFrame()

            web_url = "{}/{}".format(web_url, i) # https://www.readings.com.au/reviews, https://www.goodreads.com/book/show

            page = requests.get(web_url)
            soup = bs(page.content, "html.parser")

            title_elements = soup.findAll("h1", {"class" : re.compile("Text Text__title1")})
            if len(title_elements) == 0:
                logger.info("   {} at page {} has no data, skipping...".format(web_url, i))
                continue
            else:
                logger.info("  extracting data from {} at page {}...".format(web_url, i))
                sub_reviews_dict["title"] = title_elements[0].text

                rating_elements = soup.findAll("div", {"class" : re.compile("RatingStatistics__column")})
                sub_reviews_dict["rating"] = rating_elements[0].text

                author_elements = soup.findAll("span", {"class" : re.compile("ContributorLink__name")})
                sub_reviews_dict["book_author"] = author_elements[0].text

                sub_reviews_dict["page_number"] = i

                reviews_elements = soup.findAll("section", {"class" : re.compile("ReviewText")})
                for i, content in enumerate(reviews_elements):
                    self.reviews_content_dict[i] = content.text

                reviews_content_df = pd.DataFrame.from_dict(self.reviews_content_dict, orient="index").drop_duplicates().reset_index(drop=True)
                sub_reviews_df = sub_reviews_df.append(sub_reviews_dict, ignore_index=True)
                sub_reviews_df = pd.concat([sub_reviews_df, reviews_content_df], axis=1).ffill()
                reviews_df = reviews_df.append(sub_reviews_df)

        reviews_df = reviews_df.reset_index(drop=True)
        reviews_df = reviews_df.rename(columns={0:"reviews"})
        logger.info("  complete scraping raw data using total time of {:.2f}s...".format(time.time() - start))
        
        logger.info("Savings data to {} directory:".format(self.FILES["RAW_DATA_DIR"]))
        fname = os.path.join(self.FILES["RAW_DATA_DIR"], "raw_goodreads_reviews")
        reviews_df.to_csv("{}.csv".format(fname))
        reviews_df.to_parquet("{}.parquet".format(fname))
        
        return reviews_df
    
    
    def readings_web_crawler(self, web_url="https://www.readings.com.au/reviews", low_range=1, high_range=550):
        start = time.time()
        reviews_df = pd.DataFrame()

        logger.info("Initiating Reviews Scraping from readings.au:")
        for i in range(low_range, high_range):
            logger.info("  extracting data from page {}...".format(i))

            if i == 1:
                web_url = web_url # https://www.readings.com.au/reviews
            else:
                web_url = "{}?page={}".format(web_url, i) # https://www.readings.com.au/reviews

            page = requests.get(web_url)
            soup = bs(page.content, "html.parser")
            reviews_elements = soup.findAll("div", {"class" : re.compile("review-index-summary review-index-summary--odd review-index-summary--*")})

            for content in reviews_elements:
                sub_reviews_dict = {}
                sub_reviews_df = pd.DataFrame()

                # # extracting reviews
                reviews_main = content.find_all("div", {"class": "review-index-summary__content copy"})
                review_links = [item.find("a").get("href") for item in reviews_main]
                temp_book = review_links[0].split("/")[2]
                temp_link = "https://www.readings.com.au/reviews/{}".format(temp_book)

                review_page = requests.get(temp_link)
                soup = bs(review_page.content, "html.parser")

                reviews_content = soup.findAll("div", {"class" : re.compile("review-post__content copy")})
                content_elements = [content.find_all("p") for content in reviews_content]
                try:
                    book_title_elements = soup.select("h1.review-post__title")[0].text
                    temp_reviews = " ".join([p.get_text() for p in content_elements[0]])
                    temp_book_author = review_links[0].split("/")[2].replace("-", " ")
                except:
                    logger.info("  > there are issue in extracting data from page {}...".format(i))
                    book_title_elements = np.nan
                    temp_reviews = np.nan
                    temp_book_author = np.nan

                sub_reviews_dict["page_number"] = i
                sub_reviews_dict["book_author"] = book_title_elements
                sub_reviews_dict["reviews"] = temp_reviews

                sub_reviews_df = sub_reviews_df.append(sub_reviews_dict, ignore_index=True)
                sub_reviews_df = sub_reviews_df.reset_index(drop=True)
                reviews_df = reviews_df.append(sub_reviews_df)

        reviews_df = reviews_df.reset_index(drop=True)
        logger.info("  complete scraping raw data using total time of {:.2f}s...".format(time.time() - start))

        logger.info("Savings data to {} directory:".format(self.FILES["RAW_DATA_DIR"]))
        fname = os.path.join(self.FILES["RAW_DATA_DIR"], "raw_reviews")
        reviews_df.to_csv("{}.csv".format(fname))
        reviews_df.to_parquet("{}.parquet".format(fname))

        logger.info("done...")
        
        return reviews_df