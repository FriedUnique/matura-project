import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import yfinance as yf
import pandas as pd
import numpy as np
import ta

import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import linregress

class Line:
    def __init__(self, df, name: str = "line"):
        self.name = name
        self.df = df

    def Value(self):
        return self.df

    def Plot(self):
        plt.plot(self.df, label=self.name)
        plt.show()

    def Slope(self, window: int):
        slopes = []
        for i in range(len(self.df) - window + 1):
            x = range(len(self.df))
            y = self.df.values
            slope, _, _, _, _ = linregress(x, y)

        return Line(pd.Series(slopes))

    def SlopeAtIndexStr(self, index, window=1):
        idx = np.searchsorted(self.df.index.astype(str), index)
        prev = self.df[max(0, idx-window)]
        cur = self.df[idx]

        return float(cur/prev)/window

    def SlopeAtIndex(self, index, window=1):
        prev = float(self.df[max(self.df.first_valid_index(), index-window)])
        cur = float(self.df[index])

        return round(float(cur/prev), 4)

    def MA(self, window):
        return Line(self.df.rolling(window).mean(), name=self.name+"_ma")

    def Intersect(self, other: 'Line'):
        crossing = (((self.df <= other.Value()) & (self.df.shift(1) >= other.Value().shift(1))) | ((self.df >= other.Value()) & (self.df.shift(1) <= other.Value().shift(1))))
        self.df = self.df.to_frame()
    
        crossIndices = []

        for index, row in self.df.iterrows():
            if crossing[index]:
                
                crossIndices.append(str(index))

        return crossIndices

def ConvertDate(stringDate: str):
    stringDate = stringDate.split(" ")[0]
    dateVals = stringDate.split("-")

    y = dateVals[0]
    m = dateVals[1]
    d = dateVals[2]

    date = f"{y}-{m}-{d}"
    date_format = "%Y-%m-%d"
    date_obj = datetime.strptime(date, date_format)

    return date_obj



class MACD:
    def __init__(self, macd: Line, signal: Line, hist: Line, ta: 'TechnicalAnalysis'):

        self.macd = macd
        self.signal = signal
        self.hist = hist

        self.ta = ta


        # lines
        self.rsiLine = self.ta.RSI()
        self.rsiLine.df.index = self.rsiLine.df.index.astype(str)

        self.rsiMaLine = self.rsiLine.MA(14)

        self.volumeLine = self.ta.V_ROC(26, 20)
        self.volumeLine.df.index = self.volumeLine.df.index.astype(str)


    def Evaluate(self):
        cross = self.macd.Intersect(self.signal)

        d = self.macd.Value()
        d.index = d.index.astype(str)

        dates = [ConvertDate(idx) for idx in list(d.index)]
        #dates = []
        trends = []

        i = 0
        for index, row in d.iterrows():
            if index in cross:
                dates.append(ConvertDate(index))
                macdVal = row
                trend = self.Signal(cross[i], float(macdVal))
                trends.append(trend)

                i += 1
                continue

            trends.append(0)

        df = pd.DataFrame(zip(dates, trends), columns=["Dates", "Trend"])

        return df


    def Signal(self, index, macdVal):
        volumeVal = float(self.volumeLine.Value()[index])
        rsiVal = float(self.rsiLine.Value()[index])

        trendSum = 0
        # distance from rsi ma to rsiVal
        

        # above zero line
        if macdVal > 0 and volumeVal < 0:
            # bearish 
            if self.rsiMaLine.Value()[index] > rsiVal:
                # downwards trend -> good
                trendSum -= 0.5

            if self.rsiLine.SlopeAtIndexStr(index, 3) < 0.8:
                trendSum -= 0.4

            if rsiVal > 65:
                trendSum -= 0.1


            return trendSum

        elif macdVal < 0 and volumeVal > 0:
            if self.rsiMaLine.Value()[index] < rsiVal:
                # upwards trend -> good
                trendSum += 0.5

            if self.rsiLine.SlopeAtIndexStr(index, 3) > 1.1:
                trendSum += 0.4

            if rsiVal < 40:
                trendSum += 0.1


            return trendSum

        # false indication
        return 0


class TechnicalAnalysis:
    def __init__(self, ticker: str, period="1y"):
        self.ticker = ticker

        self.data = self.DownloadData("AAPL", period)

    
    def DownloadData(self, ticker: str, period="1y"):
        data = yf.Ticker(ticker=ticker).history(period)
        return data

    def V_ROC(self, window=26, smooth=20):
        wma = ta.trend.wma_indicator(ta.momentum.ROCIndicator(self.data["Volume"], window=window).roc(), smooth)
        return Line(wma, "VROC")

    def RSI(self):
        return Line(ta.momentum.RSIIndicator(self.data["Close"]).rsi(), "RSI")

    def ROC(self):
        return Line(ta.momentum.ROCIndicator(self.data["Close"]).roc(), "ROC")

    def MACD(self):
        ema_12 = self.data["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = self.data["Close"].ewm(span=26, adjust=False).mean()

        macd = (ema_12 - ema_26).rename("MACD")
        signal = macd.ewm(span=9, adjust=False).mean().rename("Signal")
        hist = (macd - signal).rename("Histogram")

        return MACD(Line(macd, "MACD"), Line(signal, "SIGNAL"), Line(hist, "HIST"), self)

    def MA(self, days):
        return Line(self.data["Close"].rolling(window=days).mean(), "MA")

    def Momentum(self, days):
        nDaysPrior = self.data["Close"].shift(-days)
        return Line(self.data["Close"] - nDaysPrior, "Momentum")
