import json
from datetime import datetime
import pandas as pd
import pickle


FINAL_PATH = lambda name: f"Results/FinalData/{name}"
RAW_PATH = lambda name: f"Results/RawData/{name}"
MODELS_PATH = lambda name: f"Results/Models/{name}"


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

def Normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def GetPeaks(data):
    peaks = []
    peak_values = []

    # Iterate over the data points and find the peaks
    for i in range(data.first_valid_index()+1, len(data) - 1):
        if (data[i] > data[i - 1] and data[i] > data[i + 1]) or (data[i] < data[i - 1] and data[i] < data[i + 1]):
            peaks.append(i)
            peak_values.append(data[i])


    # Create a new DataFrame to store the peaks and their values
    peaks_df = pd.DataFrame({'Peak Index': peaks, 'Peak Value': peak_values})

    return peaks_df

def GetTickerInfo():
    # names, lastFetch
    with open("tickers.json") as jFile:
        data = json.load(jFile)

    return data

def UpdateLastFetch(name: str, date: str):
    data = GetTickerInfo()
    data["lastFetch"][name] = date

    with open("tickers.json", "w") as jFile:
        json.dump(data, jFile)



def LoadModel(filename: str):
    modelInfoName = filename.split(".sav")[0] + "_info.json"
    data = None
    with open(MODELS_PATH(modelInfoName), "r") as jFile:
        data = json.load(jFile)

    return pickle.load(open(MODELS_PATH(filename), 'rb')), data