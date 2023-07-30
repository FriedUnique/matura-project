import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

import requests
from bs4 import BeautifulSoup

import os
from typing import List, Dict
from datetime import datetime
from time import sleep

import pandas as pd

from .NewsArticle import Article, ArticleSentiment

MONTHS = {"January": 1, "February": 2, "March": 3, 
          "April": 4, "May": 5, "June": 6, 
          "July": 7, "August": 8, "September": 9, 
          "October": 10, "November": 11, "December": 12}


progressDict = {}


def UpdateProgess(key, val):
    progressDict[key] = val

    result = ""
    for key, val in progressDict.items():
        result += f"[Scraper Progress {key}] {val}\n"

    print(result, end="\r")



class Day:
    def __init__(self, date: datetime):
        self.date = date

        self.totalSentiment = 0

        self.avgSentiment = 0
        self.sentimentCount = 0

    def AddSentimentValue(self, sentimentValue):
        self.sentimentCount += 1
        self.totalSentiment += sentimentValue

    def ComputeDay(self):
        # sentiment
        if self.sentimentCount == 0: return

        self.avgSentiment = round(self.totalSentiment / self.sentimentCount, 2)


class ScrapeResult:
    def __init__(self, resultParagraphs: List[Article]):
        self.articles = resultParagraphs
        self.amountOfArticlesPerDay = self.CountArticlesPerDay(self.articles)


    def GetDays(self):
        df = self.ToDataFrame()
        return self.SentimentPerDate(df)

    def CountArticlesPerDay(self, articlesList: List[Article]):
        perDay = {}
        for article in articlesList:
            if article.date in perDay.keys():
                perDay[article.date] += 1
                continue

            perDay[article.date] = 1

        return perDay

    def SortDict(self, d, isDescending = False): return dict(sorted(d.items(), reverse = isDescending))


    def SentimentPerDate(self, dataFrame: pd.DataFrame) -> Dict[str, Day]:
        avgValues: Dict[str, Day] = {}

        for i in range(len(dataFrame)):
            if dataFrame["Dates"][i] in avgValues:
                avgValues[dataFrame["Dates"][i]].AddSentimentValue(dataFrame["Sentiment"][i])
                continue

            avgValues[dataFrame["Dates"][i]] = Day(dataFrame["Dates"][i])
            avgValues[dataFrame["Dates"][i]].AddSentimentValue(dataFrame["Sentiment"][i])


        for key in list(avgValues.keys()):
            avgValues[key].ComputeDay()

        return self.SortDict(avgValues)

    def ToDataFrame(self) -> pd.DataFrame:
        dates = [article.date for article in self.articles]
        sent = [article.sentiment.raw for article in self.articles]
        articles = [f"{article.date}{article.title}|{article.body}|{article.sentiment.raw}" for article in self.articles]
        df = pd.DataFrame(zip(dates, sent, articles), columns=["Dates", "Sentiment", "Article"])
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

        # maybe the webdriver is not needed, because the site is not dynamicly generated
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
        print(f"[Scraper {self.query}]: Finding News Articles.")

        pages: List[str] = []

        for n in range(n_pages):
            # load the search page
            self.url = self.BuildURL(self.query, self.dateRange, self.section, n)
            self.driver.get(self.url)

            sleep(1) # so the driver can load the webpage fully

            try:
                ul_element = self.driver.find_element(By.CSS_SELECTOR, "ul.search-results__list__2SxSK")
                li_elements = ul_element.find_elements(By.CSS_SELECTOR, "li")
            except NoSuchElementException:
                #print(f"[Scraper {self.query}]: Found All Pages in Range")
                break

            for li in li_elements:
                link = li.find_element(By.CSS_SELECTOR, "a").get_attribute("href")

                # remove duplicates
                if link in pages:
                    continue
                pages.append(link)

        print(f"[Scraper {self.query}]: Found ", len(pages), " Articles.")
        return pages

    def GetPageResults(self, pages: List[str]) -> ScrapeResult:
        #print(f"[Scraper {self.query}]: Starting To Collect News Data From Articles.")
        print()
        
        articles: List[Article] = []
        readArticles = 0
        for page in pages:

            readArticles += 1

            pageResponce = requests.get(page)
            soup = BeautifulSoup(pageResponce.text, "html.parser")

            pageHeaderElement = soup.select_one("div.article-header__heading__15OpQ")
            

            if pageHeaderElement:
                title = pageHeaderElement.select_one("h1").text
                date = pageHeaderElement.select_one("span.date-line__date__23Ge-").text

                # continue to next page, because podcasts have no use to us.
                if str(title).startswith("Podcast: "):
                    continue

                articleElement = soup.select_one("div.article-body__content__17Yit")

                if not articleElement:
                    continue

                paragraphs = articleElement.select("p")
                articleBody = " ".join([p.text for p in paragraphs])

                sent = self.analyzer.GetSentiment(title + articleBody)
                a = Article(self.ConvertDate(date), title, articleBody, ArticleSentiment(sent, self.analyzer.SentimentEval(sent)))
                articles.append(a)

            #UpdateProgess(self.query, f"Articles Read = {readArticles}/{len(pages)}")
            #print(f"[Scraper]: Articles Read = {readArticles}/{len(pages)}", end="\r")

        return ScrapeResult(articles)

    def ConvertDate(self, stringDate: str):
        stringDate = stringDate.replace(",", "")
        m = stringDate.split(" ")[0]
        d = stringDate.split(" ")[1]
        y = stringDate.split(" ")[2]

        date = f"{y}-{MONTHS[m]}-{d}"
        date_format = "%Y-%m-%d"
        date_obj = datetime.strptime(date, date_format)

        return str(date_obj).split(" ")[0]

  
class SentimentAnalyzer:
    # nltk.download("vader_lexicon")
    def __init__(self):
        #self.Download()

        self.sia = SentimentIntensityAnalyzer()

    def Download(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('vader_lexicon')

    def Preprocess(self, text: str):
        # Tokenize the text into words
        tokens = word_tokenize(text.lower())

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words]

        # Join the filtered tokens back into a single string
        preprocessed_text = ' '.join(filtered_tokens)
    
        return preprocessed_text

    def GetSentiment(self, text: str) -> float:
        sentiment_scores = self.sia.polarity_scores(self.Preprocess(text))
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

