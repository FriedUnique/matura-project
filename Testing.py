import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#pd.set_option("display.max_columns", 10)
pd.set_option('display.width', 800)
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def ACF(series):
    acf_result = [series.autocorr(lag=i) for i in range(41)]  # Adjust the range as needed

    # Plot the ACF
    plt.figure(figsize=(12, 6))
    plt.stem(acf_result, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lag')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True)
    plt.show()

def CorrelationMatrix(df):
    correlation_matrix = df.corr()

    # Create a heatmap using Matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar(label='Correlation Coefficient')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title('Correlation Heatmap using Matplotlib')
    plt.show()

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

def FillNonTradingDays(series, originalDates, compareDates):
    """
    originalDates: the dates of the series
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

def Rsi(close, days=14):
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


def getData():
    ohlc = yf.Ticker("AAPL").history("3y")
    dates = [ConvertDate(idx) for idx in list(ohlc.index)]
    sent = pd.read_csv("Apple.csv")
    sent = sent.drop("Unnamed: 0", axis=1)

    sentDates = sent["Date"]
    o = FillNonTradingDays(ohlc["Open"], dates, sentDates)
    h = FillNonTradingDays(ohlc["High"], dates, sentDates)
    l = FillNonTradingDays(ohlc["Low"], dates, sentDates)
    c = FillNonTradingDays(ohlc["Close"], dates, sentDates)
    v = FillNonTradingDays(ohlc["Volume"], dates, sentDates)

    df = pd.DataFrame(zip(o, h, l, c), sentDates, columns=["Open", "High", "Low", "Close"]).reset_index()

    # scale OHLC
    # scaler = MinMaxScaler()
    # normOHLC = pd.DataFrame(scaler.fit_transform(df[["Open", "High", "Low", "Close"]]), columns=["Open", "High", "Low", "Close"])
    # df = pd.concat([df.drop(["Open", "High", "Low", "Close"], axis=1), normOHLC], axis=1)

    df["Close_Lag_1"] = c.shift(1)
    df["Close_Lag_3"] = c.shift(3)
    df["Close_Lag_7"] = c.shift(7)
    #df["Close_Lag_14"]

    df["Positive %"] = sent["Positive"] / sent["Article Count"]
    df["Postive % avg"] = df["Positive %"].rolling(14).mean()
    df["Sentiment"] = sent["AverageSentiment"]
    df["Average Sentiment 14"] = sent["AverageSentiment"].rolling(14).mean()
    df["Article Count"] = sent["Article Count"]
    df["Article Count 14"] = sent["Article Count"].rolling(14).mean().apply(lambda x : round(x, 0))

    df['AvgSentiment_Lag_1'] = df['Average Sentiment 14'].shift(1)
    df['AvgSentiment_Lag_3'] = df['Average Sentiment 14'].shift(3)
    df['AvgSentiment_Lag_7'] = df['Average Sentiment 14'].shift(7)

    df["RSI"] = Rsi(df["Close"])

    df["Direction 14"] = (c.pct_change(14) * 100 > 0).astype(int)


    dates = df["Date"]
    #df = df.drop("Date", axis=1)
    df = df.dropna()

    #CorrelationMatrix(df)
    #ACF(df["Average Sentiment 14"])

    return df, dates


    

def LALAL():
    dataFrame = getData()
    labels = dataFrame["Direction 14"]
    dataFrame = dataFrame.drop("Direction 14", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(dataFrame, labels, test_size=0.2, random_state=42)

    X_train = X_train.reshape([-1, 28, 28]).astype("float32")
    X_test = X_test.reshape([-1, 28, 28]).astype("float32")

    print(X_train.shape[0])

    model = keras.Sequential()
    model.add(keras.Input(shape=(None, 16)))
    model.add(layers.LSTM(256, return_sequences=True, activation="relu"))
    model.add(layers.LSTM(256, name="lstm_layer2"))
    model.add(layers.Dense(10))

    print(model.summary())
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=2)
    model.evaluate(X_test, y_test, batch_size=64, verbose=2)

def cool():
    df, dates = getData()
    dates = pd.to_datetime(dates)
    ohlc = df[["Open", "High", "Low", "Close", "Sentiment"]]

    scaler = StandardScaler()
    dfScaledTraining = scaler.fit_transform(ohlc)

    trainX = []
    trainY = []

    n_future = 30 # days to predict in the future
    n_past = 14  # days to base this prediction on in the past

    # (n_samples x timesteps x n_features)
    for i in range(n_past, len(df) - n_future +1):
        trainX.append(dfScaledTraining[i - n_past:i, 0:dfScaledTraining.shape[1]])
        trainY.append(dfScaledTraining[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)
    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))


    model = keras.Sequential()
    model.add(layers.LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(layers.LSTM(32, activation='relu', return_sequences=False))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(trainY.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)

    forecastDates = pd.date_range(list(dates)[-1], periods=n_future, freq="1d").tolist()
    forecast = model.predict(trainX[-n_future:])

    yPredReal = scaler.inverse_transform(np.repeat(forecast, dfScaledTraining.shape[1], axis=-1))[:,0]

    print()
    print("Date From: ", list(dates)[-1])
    print("Date To: ", list(dates)[-1] + timedelta(days=n_future))
    print()

    forecast_dates = []
    for i in forecastDates:
        forecast_dates.append(i.date())

    df_forecast = pd.DataFrame({"Date": np.array(forecast_dates), "Open":yPredReal})
    df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])

    original = df[["Date", "Open"]]
    original["Date"] = pd.to_datetime(dates)

    plt.plot(original["Date"].values, original["Open"].values)
    plt.plot(df_forecast["Date"].values, df_forecast["Open"].values)
    plt.show()



cool()




# fig, axs = plt.subplots(4, sharex=True, sharey=False)

# # axs[0].set_title("Positive %")
# axs[0].plot(fin["Positive %"].rolling(30).mean(), color="red")

# axs[2].bar(fin["Article Count"].index, fin["Article Count"])
# axs[3].plot(fin["Close"].rolling(7).mean())

# #ax2.plot(fin["Close"].rolling(7).mean())

# axs[1].plot(Rsi(fin["Close"]), color="orange", linewidth=1)

# # Oversold
# axs[1].axhline(30, linestyle='--', linewidth=1.5, color='green')
# # Overbought
# axs[1].axhline(70, linestyle='--', linewidth=1.5, color='red')

# plt.show()

"""
direction14 = (c.pct_change(14) * 100 > 0).astype(int)
    #direction14 = direction14.apply(lambda x: 1 if x > 1 else 0)
    direction14.name = "Direction"

    cAverage14 = c.rolling(14).mean()

    positivePercentage = sent["Positive"] / sent["Article Count"]
    positivePercentage.name = "Positive %"

    positivePercentage14avg = positivePercentage.rolling(14).mean()
    positivePercentage14avg.name = "Postive % avg"

    averageSentiment = sent["AverageSentiment"]
    averageSentiment.name = "Average Sentiment"

    averageSentiment14avg = averageSentiment.rolling(14).mean()
    averageSentiment14avg.name = "Average Sentiment 14"

    articleCount = sent["Article Count"]
    
    articleCount14 = articleCount.rolling(14).mean().apply(lambda x : round(x, 0))
    articleCount14.name = "Article Count 14"

"""