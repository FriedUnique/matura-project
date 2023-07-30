import Helper

import yfinance
import pandas as pd



class Computer:
    def __init__(self):

        tickerInfo = Helper.GetTickerInfo()
        tickerDict = dict(tickerInfo["tickers"])

        listOfDfs = []

        for ticker in tickerDict.keys():
            try:
                listOfDfs.append(self.CompileCompanyDf(ticker, 1))
            except FileNotFoundError as e:
                print(e)

        self.dataframe = pd.concat(listOfDfs)

        return self.dataframe
        
    def GetOHLC(self, ticker: str, sentDates: list):
        ohlc = yfinance.Ticker(ticker).history("3y")
        dates = [Helper.ConvertDate(idx) for idx in list(ohlc.index)]

        precision = 3

        o = Helper.FillNonTradingDays(ohlc["Open"], dates, sentDates)
        h = Helper.FillNonTradingDays(ohlc["High"], dates, sentDates)
        l = Helper.FillNonTradingDays(ohlc["Low"], dates, sentDates)
        c = Helper.FillNonTradingDays(ohlc["Close"], dates, sentDates)

        return pd.DataFrame(zip(o, h, l, c), sentDates, columns=["Open", "High", "Low", "Close"]).apply(lambda x : round(x, precision))

    def GetSentiment(self, name: str, maWindow: int):
        sentiment = pd.read_csv(Helper.FINAL_PATH(f"{name}_finalDataframe.csv")).drop("Unnamed: 0", axis=1)
        return sentiment["Sentiment"].rolling(window=maWindow).mean().fillna(0), sentiment["Dates"], sentiment["Signal"]

    def Trend(self, ohlc: pd.DataFrame, days: int):
        # one hot encoding
        trend = ohlc["Close"].shift(-days) / ohlc["Close"]
        trend.name = "Label"

        pos, neu, neg = [], [], []
        margins = 0.01

        for idx, val in trend.items():
            if val > 1+margins:
                pos.append(1)
                neu.append(0)
                neg.append(0)

            elif val < 1-margins:
                pos.append(0)
                neu.append(0)
                neg.append(1)

            else:
                pos.append(0)
                neu.append(1)
                neg.append(0)

        #trend = pd.DataFrame(zip(pos, neu, neg), columns=["+Trend", "0Trend", "-Trend"], index=trend.index)
        return trend

        
    def CompileCompanyDf(self, companyName: str, sentAvgDays = 1):

        sent, sentDates, _ = self.GetSentiment(companyName, sentAvgDays)
        ohlc = self.GetOHLC(self.tickerDict[companyName], list(sentDates))

        sentimentSeries = sent
        sentimentSeries.index = sentDates

        prevClose = []
        prevSentiment = []

        for day in range(self.memoryDepth):
            prevClose.append(pd.Series(ohlc["Close"].shift(day+1), name=f"Prev Close {day+1}"))
            prevSentiment.append(pd.Series(sentimentSeries.shift(day+1), name=f"Prev Sent {day+1}"))

        df = pd.concat([ohlc,
                        sentimentSeries,
                        pd.concat(prevClose, axis=1), 
                        pd.concat(prevSentiment, axis=1),
                        self.Trend(ohlc, self.futureDistance)
                        ], axis=1)

        #self.Trend(ohlc, 5)
        """clean the data"""
        df.fillna(0, inplace=True)
        df = df[:-self.futureDistance]       

        return df