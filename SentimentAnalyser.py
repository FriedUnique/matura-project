import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from datetime import datetime, date
from typing import Dict
import pandas as pd


class Day:
    def __init__(self, date: date):
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

    def SentimentEval(self, rawValue: float):
        if rawValue >= 0.05:
            return 1
        elif rawValue <= -0.05:
            return -1
        else:
            return 0
        


def Main(fileName: str, dateFormat: str):
    sa = SentimentAnalyzer()

    days: Dict[datetime, Day] = {}
    lines = []
    with open(fileName, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line == "":
            continue

        dateString, title, body = line.split("|")
        dateString = dateString.replace(" ", "")

        date = datetime.strptime(dateString, dateFormat).date()
        article = title + body

        ppText = sa.Preprocess(article)

        if date in list(days.keys()):
            days[date].AddSentimentValue(sa.GetSentiment(ppText))
        else:
            d = Day(date)
            d.AddSentimentValue(sa.GetSentiment(ppText))
            days[date] = d


    for day in list(days.values()):
        day.ComputeDay()

    dates = [day.date for day in list(days.values())]
    sentiment = [day.avgSentiment for day in list(days.values())]

    pd.DataFrame(zip(dates, sentiment, [0 for i in range(len(sentiment))]), columns=["Dates", "Sentiment", "Signal"]).to_csv("./Data/NewAnalysis.csv")

# %B,%d,%Y
Main("articles.txt", "%m,%d,%Y")
