import Sentiment
from TradingAnalysis import TechnicalAnalysis
from Helper import GetTickerInfo, UpdateLastFetch, FINAL_PATH, RAW_PATH

import pandas as pd
from datetime import datetime

from typing import Dict, List



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


class SentimentEval:
    def __init__(self, name: str, df: pd.DataFrame):
        self.name = name
        perDate = self.SentimentPerDate(df)

        today = str(datetime.today()).split(" ")[0]


        dates = list(perDate.keys())
        sent = [s.avgSentiment for s in list(perDate.values())]
        # not really needed ...
        signals = self.TradingAnalysis(GetTickerInfo()["tickers"][name.lower()], dates, "3y")

        self.df = pd.DataFrame(zip(dates, sent, signals), columns=["Dates", "Sentiment", "Signal"])

        try:
            # new, old
            self.df = self.MergeDataframe(self.df, self.Read(FINAL_PATH(f"{name}_finalDataframe.csv")))
        except FileNotFoundError:
            print("FileNotFoundError")
        
        self.Save()
        UpdateLastFetch(name, today)

    @classmethod
    def Read_CSV(cls, csvFile: str):
        """csvFile is the query_rawdata.csv in the Results folder"""
        df = pd.read_csv(csvFile)
        name = csvFile.split("/")[-1].split("_")[0]

        return cls(name, df)

    @classmethod
    def AnalyzeSentiment(cls, query: str, dateRange: Sentiment.Article.DateRange, section: Sentiment.Article.Section, pages=1):
        scraper = Sentiment.Scraper(query, dateRange, section)

        pagesOfInterest = scraper.GetNewsPages(pages)
        scrapeResult = scraper.GetPageResults(pagesOfInterest)

        df = scrapeResult.ToDataFrame()
        print(f"[Scraper {query}] Saved Raw Data")
        df.to_csv(RAW_PATH(f"{query}_rawdata.csv"))

        return cls(query, df)


    def TradingAnalysis(cls, ticker: str, existingDates: List[str], period = "1y") -> List[float]:
        ta = TechnicalAnalysis(ticker, "1y")
        trends = []
        a = ta.MACD().Evaluate()

        for date in existingDates:
            trend = list(a[a["Dates"] == date]["Trend"])

            # if the date does not exist
            if len(trend) == 0:
                trends.append(0)
                continue

            trends.append(trend[0])

        return trends




    def Read(self, path):
        return pd.read_csv(path)

    def Save(self, name = "finalDataframe.csv"):
        self.df.to_csv(FINAL_PATH(f"{self.name}_{name}"))

    def SortDict(self, d, isDescending = False): return dict(sorted(d.items(), reverse = isDescending))

    def MergeDataframe(self, new, old):
        test = pd.concat([new, old])
        test = test.sort_values("Dates").drop_duplicates("Dates")
        test.reset_index(drop=True, inplace=True)
        return test.drop("Unnamed: 0", axis=1)


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