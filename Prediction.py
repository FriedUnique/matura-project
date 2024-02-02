import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os

import yfinance

from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import tabloo
import seaborn as sns

def CorrelationMatrix(df):
    correlation_matrix = df.corr()

    # Create a heatmap using Seaborn
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 13})
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, square=True)

    plt.title('Correlation Heatmap')
    plt.show()



def FillNonTradingDays(series, originalDates, compareDates):
    """originalDates: the dates of the series
    compareDates: the dates, which 'edit' the series
    """
    s = pd.Series(series.values, originalDates)
    prev = pd.NA
    vals = []
        
    for idx in compareDates:
        try:
            prev = s[idx]
            vals.append(s[idx])
        except KeyError:
            vals.append(prev)

    return pd.Series(vals).dropna()

def ConvertDate(stringDate: str):
    stringDate = str(stringDate)
    stringDate = stringDate.split(" ")[0]
    dateVals = stringDate.split("-")

    y = dateVals[0]
    m = dateVals[1]
    d = dateVals[2]

    date = f"{y}-{m}-{d}"
    date_format = "%Y-%m-%d"
    date_obj = datetime.strptime(date, date_format)

    return date_obj


def GetOHLC(ticker: str, sentDates: list):
        ohlc = yfinance.Ticker(ticker).history("3y")
        dates = [ConvertDate(idx) for idx in list(ohlc.index)]

        o = FillNonTradingDays(ohlc["Open"], dates, sentDates)
        h = FillNonTradingDays(ohlc["High"], dates, sentDates)
        l = FillNonTradingDays(ohlc["Low"], dates, sentDates)
        c = FillNonTradingDays(ohlc["Close"], dates, sentDates)

        return pd.DataFrame(zip(o, h, l, c), sentDates, columns=["Open", "High", "Low", "Close"]).apply(lambda x : round(x, 3))


def RSI(close, days=14) -> pd.Series:
    change = close.diff()
    change.dropna(inplace=True)

    change_up = change.copy()
    change_down = change.copy()

    change_up[change_up<0] = 0
    change_down[change_down>0] = 0
    change.equals(change_up+change_down)

    avg_up = change_up.rolling(days).mean()
    avg_down = change_down.rolling(days).mean().abs()

    rsi = 100 * avg_up / (avg_up + avg_down)
    rsi.name = "rsi"
    return rsi


def DataframeForEach(fileName: str, ticker: str, futureDistance: int, pastViewDistance: int) -> pd.DataFrame:
    """futureDistance: how far in the future the label sits or rather how far in the future the model should predict
       pastViewDistance: on what data the future prediction should be based on
    """

    sentimentData = pd.read_csv(fileName).drop("Unnamed: 0", axis=1)
    sentimentDates = sentimentData["Dates"]
    sentiment = sentimentData["Sentiment"]
    sentiment.index = sentimentDates

    dataframe = GetOHLC(ticker, sentimentDates)#.rolling(7).mean()

    dataframe["MA50"] = dataframe["Close"].rolling(50).mean()
    #dataframe["MA100"] = dataframe["Close"].rolling(100).mean()

    dataframe["Close_Lag_1"] = dataframe["Close"].shift(1)
    dataframe["Close_Lag_3"] = dataframe["Close"].shift(3)
    dataframe["Close_Lag_7"] = dataframe["Close"].shift(7)


    #* overbought v underbought expressed via rsi

    rsi = RSI(dataframe["Close"])#.rolling(pastViewDistance).mean()

    dataframe["Overbought"] = rsi / 70
    dataframe["Oversold"] = 30/rsi

    #* sentiment

    dataframe["Sentiment"] = sentiment
    dataframe["Average Sentiment"] = sentiment.rolling(pastViewDistance).mean() 

    dataframe['AvgSentiment_Lag_7'] = dataframe["Average Sentiment"].shift(7)
    dataframe['AvgSentiment_Lag_14'] = dataframe["Average Sentiment"].shift(14)

    #dataframe["Label"] = (dataframe["Close"].shift(30).pct_change(30) * 100>0).astype(int).rolling(14).mean()
    dataframe["Label"]  = (dataframe["Close"].shift(-30)/dataframe["Close"] > 1).astype(int)
    #dataframe["Label"] = (dataframe["Close"].shift(-30)/dataframe["Close"])
    #dataframe["Label"] = (dataframe["Close"].shift(futureDistance).pct_change(futureDistance) * 100).astype(float)
    #dataframe["LabelTest"] = (dataframe["Close"].shift(futureDistance).pct_change(futureDistance) * 100 > 0).astype(int)#.rolling(30).mean().shift(14)

    return dataframe.dropna()

def CreateDataframe(futurePredictionDistance: int, pastMemoryDepth: int):

    tickerDic = {"apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "intel": "INTC", "tesla": "TSLA", "meta": "META", "alphabet": "GOOG"}

    bigDataframe = pd.DataFrame()

    for fileName in os.listdir("./Data"):
        company = fileName.lower().split(".")[0]
        if company in tickerDic:
            df = DataframeForEach(f"./Data/{fileName}", tickerDic[company], futurePredictionDistance, pastMemoryDepth)
            bigDataframe = pd.concat([bigDataframe, df])

    return bigDataframe


def DataframeForPrediction(fileName: str, ticker: str, futureDistance: int, pastViewDistance: int):
    sentimentData = pd.read_csv(fileName).drop("Unnamed: 0", axis=1)
    sentimentDates = sentimentData["Dates"]
    sentiment = sentimentData["Sentiment"]
    sentiment.index = sentimentDates

    dataframe = GetOHLC(ticker, sentimentDates)#.rolling(7).mean()

    dataframe["MA50"] = dataframe["Close"].rolling(50).mean()
    #dataframe["MA100"] = dataframe["Close"].rolling(100).mean()

    dataframe["Close_Lag_1"] = dataframe["Close"].shift(1)
    dataframe["Close_Lag_3"] = dataframe["Close"].shift(3)
    dataframe["Close_Lag_7"] = dataframe["Close"].shift(7)


    #* overbought v underbought expressed via rsi

    rsi = RSI(dataframe["Close"])#.rolling(pastViewDistance).mean()

    dataframe["Overbought"] = rsi / 70
    dataframe["Oversold"] = 30/rsi

    #* sentiment

    dataframe["Sentiment"] = sentiment
    dataframe["Average Sentiment"] = sentiment.rolling(pastViewDistance).mean() 

    dataframe['AvgSentiment_Lag_7'] = dataframe["Average Sentiment"].shift(7)
    dataframe['AvgSentiment_Lag_14'] = dataframe["Average Sentiment"].shift(14)

    return dataframe



def TrainModel(n_future: int, features):
    data = CreateDataframe(n_future, 14)
    #tabloo.show(data)

    data = data.reset_index()[features]



    label = data[f"Label"]

    data = data.drop(f"Label", axis=1)

    print(label)


    # split into train and test sets
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Reshape
    data_reshaped = np.reshape(data_scaled, (data_scaled.shape[0], 1, data_scaled.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(data_reshaped, label, test_size=0.2, random_state=42)


    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))  # Adjust activation based on your problem (e.g., 'softmax' for multi-class classification)

    optimizer = keras.optimizers.Adam(lr=0.01)
    # mae, mse, bce
    model.compile(optimizer=optimizer, loss='mse')  # Use appropriate loss function for your problem

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate on the test set

    plt.plot(pd.Series(history.history['loss']))
    plt.plot(pd.Series(history.history['val_loss']))
    plt.title('Modellfehler (MSE)')
    plt.ylabel('Fehler')
    plt.xlabel('Epoche')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # model.save("model1.h5")

    return model, scaler


def TestModel(n_future: int, features: list, model, scaler):
    data = DataframeForPrediction("./Data/Apple.csv", "AAPL", n_future, 14)
    #print(data.index)

    features.remove("Label")

    dates = data.index
    print(dates[-1])
    data = data.reset_index()[features].iloc[-30:]

    data_scaled = scaler.transform(data)

    # Reshape
    data_reshaped = np.reshape(data_scaled, (data_scaled.shape[0], 1, data_scaled.shape[1]))

    forecast = model.predict(data_reshaped)
    ma = pd.DataFrame(forecast).rolling(5).mean()

    fig, (ax1, ax2)  = plt.subplots(2, sharex=True)

    ax1.set_title(f"Trendprognose (base(0): {dates[-1]})")
    ax1.plot(forecast)
    ax1.plot(ma)

    f = pd.DataFrame(forecast)
    m, b = np.polyfit(f.index, f.values, 1)
    #ax1.plot(f.index, f.index*m + b)

    data = yfinance.Ticker("AAPL").history("3y").reset_index()
    idx = data[data["Date"] == dates[-1]].index.item()

    ax2.plot(data.iloc[idx:idx+n_future]["Close"].reset_index(drop=True))
    ax2.set_title("Close Price In The Future")

    plt.xlabel("Börsentage")
    ax2.set_ylabel("Dollar")
    ax1.set_ylabel("Trend %")
    
    plt.show()





def TrainModelBetterMan(features, n_future, n_past):
    # n_future = 30 # days to predict in the future
    # n_past = 14  # days to base this prediction on in the past

    df = CreateDataframe(n_future, n_past) #DataframeForEach("./Data/Apple.csv", "AAPL", n_future, n_past)
    ohlc = df[features]
    print(ohlc)

    scaler = StandardScaler()
    dfScaledTraining = scaler.fit_transform(ohlc)

    print(pd.DataFrame(dfScaledTraining))

    trainX = []
    trainY = []

    # (n_samples x timesteps x n_features)
    for i in range(n_past, len(df) - n_future +1):
        trainX.append(dfScaledTraining[i - n_past:i, 1:dfScaledTraining.shape[1]])
        trainY.append(dfScaledTraining[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)
    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))


    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))

    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()

    history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

    forecast = model.predict(trainX[-n_future:])

    yPredReal = scaler.inverse_transform(np.repeat(forecast, dfScaledTraining.shape[1], axis=-1))[:,0]

    print(yPredReal[-1])

    df_forecast = pd.DataFrame({"Open":yPredReal})

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    model.save("thatsMoreLikeIt.h5")
    return model, scaler



def Test(filePath: str, ticker: str, features, n_future, n_past, scaler, model):
    # n_future = 30 # days to predict in the future
    # n_past = 14  # days to base this prediction on in the past

    df = DataframeForEach(filePath, ticker, n_future, n_past)
    ohlc = df[features]# [[f"Label{n_future}", "Open", "High", "Low", "Close", "Average Sentiment", "MA50"]

    dates = df.reset_index()["Dates"]
    dates = pd.to_datetime(dates)

    dfScaledTraining = scaler.transform(ohlc)

    trainX = []
    trainY = []

    # (n_samples x timesteps x n_features)
    for i in range(n_past, len(df) - n_future +1):
        trainX.append(dfScaledTraining[i - n_past:i, 1:dfScaledTraining.shape[1]])
        trainY.append(dfScaledTraining[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)
    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))

    #model = tf.keras.models.load_model("thatsMoreLikeIt.h5")

    forecast = model.predict(trainX[-n_future:])
    yPredReal = scaler.inverse_transform(np.repeat(forecast, dfScaledTraining.shape[1], axis=-1))[:,0]
    print(yPredReal)

    # plt.plot(df["Open"].values)

    print(dates)

    plt.plot(yPredReal, linewidth=3)
    title = plt.title(f"Trend Prediction (base: {dates.iloc[-1].date()}) (point: {dates.iloc[-1].date() + timedelta(n_future)})")

    color = "green" if yPredReal[-1] > 0.6 else "red" if yPredReal[-1] < 0.4 else "orange"

    plt.plot(n_future-1, yPredReal[-1], "ro", color=color)
    plt.show()




if __name__ == "__main__":
    future, past = 30, 14
    f = ["Label", "Open", "High", "Low", "Close", "Average Sentiment", "Overbought", "MA50", "Close_Lag_7"]
    # model, scalar = TrainModelBetterMan(f, future, past)
    # Test("./Data/Apple.csv", "AAPL", f, future, past, scalar, model)

    m, s = TrainModel(30, f)
    TestModel(30, f, m, s)

    # df = DataframeForEach("./Data/Apple.csv", "AAPL", 30, 14)
    # tabloo.show(df)

    #.save("modelVeryGoodLossDontBelieveIt.h5")

    # data = yfinance.Ticker("AAPL").history("1y").reset_index()
    # idx = data[data["Date"] == "2024-01-23"].index.item()
    # print(data.iloc[idx:idx+30])