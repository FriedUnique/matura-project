import Helper
import Evaluator

from Sentiment import Article
from DataFrameComputer import Computer

import pandas as pd
import numpy as np
import yfinance as yf

from threading import Thread

from datetime import datetime
from typing import Dict, List


class SentimentFetcher:
    def __init__(self):
        self.threads: List[Thread] = []

    @classmethod
    def ReadFromRawData(cls):
        for ticker in Helper.GetTickerInfo()["tickers"]:
            path = f"Results/RawData/{ticker}_rawdata.csv"
            Evaluator.SentimentEval.Read_CSV(path)

        return cls()

    @classmethod
    def StartThreads(cls, refreshAll=False):
        today = datetime.today()

        for ticker in Helper.GetTickerInfo()["tickers"]:
            lastDate = datetime.strptime(Helper.GetTickerInfo()["lastFetch"][ticker], "%Y-%m-%d")
            difference = (today - lastDate).days

            dateRange = Article.DateRange.Any
            if difference <= 7:
                dateRange = Article.DateRange.Week
            elif difference <= 30:
                dateRange = Article.DateRange.Month
            elif difference <= 360:
                dateRange = Article.DateRange.Year
            else:
                dateRange = Article.DateRange.Any

            if refreshAll:
                dateRange = Article.DateRange.Any

            thread = Thread(target=Evaluator.SentimentEval.AnalyzeSentiment, args=(ticker, dateRange, Article.Section.All, 100_000,))
            thread.start()
        return cls()
            


if __name__ == "__main__":
    SentimentFetcher.StartThreads(False)
    finalDataFrame = Computer()

