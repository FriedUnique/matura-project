from nltk.sentiment import SentimentIntensityAnalyzer

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

import os
from typing import List
from datetime import datetime

import pandas as pd

from .NewsArticle import Article, ArticleSentiment

MONTHS = {"January": 1, "February": 2, "March": 3, 
          "April": 4, "May": 5, "June": 6, 
          "July": 7, "August": 8, "September": 9, 
          "October": 10, "November": 11, "December": 12}

class ScrapeResult:
    def __init__(self, resultParagraphs: List[Article]):
        self.articles = resultParagraphs

    def ToDataFrame(self) -> pd.DataFrame:
        dates = [article.date for article in self.articles]
        sent = [article.sentiment.raw for article in self.articles]
        df = pd.DataFrame(zip(dates, sent), columns=["Dates", "Sentiment"])
        return df


    def Save(self):
        if not os.path.exists("Results"):
            os.mkdir("Results")

        with open(f"Results/results.csv", "w") as f:
            for article in self.articles:
                f.write(f"{article.date}{article.title}|{article.body}|{article.sentiment.raw}\n")


class Scraper:
    def __init__(self, query: str, dateRange: Article.DateRange, section = Article.Section.All): 
        self.query = query
        self.dateRange = dateRange
        self.section = section

        self.url = self.BuildURL(self.query, self.dateRange, self.section, 0)

        self.analyzer = SentimentAnalyzer()

        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--log-level=3")
        self.driver = webdriver.Chrome(chrome_options=self.chrome_options)

        #self.url = f"https://www.reuters.com/site-search/?query={self.query}&date={self.dateRange.value}&section={self.section.value}&offset=0"
        #self.simpleURL = f"https://www.reuters.com/company/{self.query}/"


    def __del__(self):
        self.driver.quit()

    def BuildURL(self, query: str, dateRange: Article.DateRange, section: Article.Section, page: int):
        return f"https://www.reuters.com/site-search/?query={query}&date={dateRange.value}&section={section.value}&offset={page*20}"
        

    def GetNewsPages(self, n_pages=1) -> List[str]:
        """Parameter 'n': Represents how many """
        print("[Scraper]: Finding News Pages.")

        pages: List[str] = []

        for n in range(n_pages):
            # load the search page
            self.url = self.BuildURL(self.query, self.dateRange, self.section, n)
            self.driver.get(self.url)

            try:
                ul_element = self.driver.find_element(By.CSS_SELECTOR, "ul.search-results__list__2SxSK")
                li_elements = ul_element.find_elements(By.CSS_SELECTOR, "li")
            except NoSuchElementException:
                break

            for li in li_elements:
                link = li.find_element(By.CSS_SELECTOR, "a").get_attribute("href")

                # remove duplicates
                if link in pages:
                    continue
                pages.append(link)

        print(len(pages))
        return pages

    def PageResults(self, pages: List[str], n=-1) -> ScrapeResult:
        print("[Scraper]: Starting To Collect News Data.")

        # pages are a string link
        articles: List[Article] = []

        # safty
        n = min(n, len(pages)-1)

        readArticles = 0
        maxLen = len(pages[:n])

        for page in pages[:n]:
            # animation to know progress
            print(f"[Scraper]: Articles Read = {readArticles}/{maxLen}", end="\r")

            self.driver.get(page)

            pageHeader = self.driver.find_element(By.CSS_SELECTOR, "div.article-header__heading__15OpQ")
            title = pageHeader.find_element(By.TAG_NAME, "h1").text
            date = pageHeader.find_element(By.CSS_SELECTOR, "span.date-line__date__23Ge-").text
            
            # continue to next page, because podcasts have no use to us.
            if str(title).startswith("Podcast: "):
                continue

            article = self.driver.find_element(By.CSS_SELECTOR, "div.article-body__content__17Yit")
            paragraphs = article.find_elements(By.TAG_NAME, "p")

            body = " ".join([p.text for p in paragraphs])

            sent = self.analyzer.GetSentiment(title + body)
            a = Article(self.ConvertDate(date), title, body, ArticleSentiment(sent, self.analyzer.SentimentEval(sent)))
            articles.append(a)

            readArticles += 1

        print()
        return ScrapeResult(articles)

    def ConvertDate(self, stringDate: str):
        stringDate = stringDate.replace(",", "")
        m = stringDate.split(" ")[0]
        d = stringDate.split(" ")[1]
        y = stringDate.split(" ")[2]

        date = f"{y}-{MONTHS[m]}-{d}"
        date_format = "%Y-%m-%d"
        date_obj = datetime.strptime(date, date_format)

        return date_obj

  
class SentimentAnalyzer:
    # nltk.download("vader_lexicon")
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def GetSentiment(self, text: str) -> float:
        sentiment_scores = self.sia.polarity_scores(text)
        return sentiment_scores['compound']

    def SentimentEval(self, rawValue: float) -> ArticleSentiment.Sentiment:
        if rawValue >= 0.05:
            return ArticleSentiment.Sentiment.Positive
        elif rawValue <= -0.05:
            return ArticleSentiment.Sentiment.Negative
        else:
            return ArticleSentiment.Sentiment.Neutral





if __name__ == "__main__":
    a = Scraper("Apple Inc", Article.DateRange.Week)
    strLinksToPages = a.GetNewsPages()
    result = a.PageResults(strLinksToPages)
