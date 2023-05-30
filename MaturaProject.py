import NSA

import pandas as pd


class SentimentEval:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        perDate = self.SentimentPerDate(self.df)
        dates = list(perDate.keys())
        sent = list(perDate.values())

        self.df = pd.DataFrame(zip(dates,sent), columns=["Dates", "Sentiment"])
        print(self.df)

    @classmethod
    def Read_CSV(cls, csvFile: str):
        df = pd.read_csv(csvFile)

        return cls(df)

    @classmethod
    def AnalyzeSentiment(cls, query: str, dateRange: NSA.Article.DateRange, section: NSA.Article.Section, pages=1):
        scraper = NSA.Scraper(query, dateRange, section)

        pagesOfInterest = scraper.GetNewsPages(pages)
        scrapeResult = scraper.PageResults(pagesOfInterest)

        df = scrapeResult.ToDataFrame()

        df.to_csv(f"./Results/{query}_rawdata.csv")

        return cls(df)

    def Save(self):
        self.df.to_csv("./dataframe.csv")

    def SentimentPerDate(self, dataFrame: pd.DataFrame):
        avgValues = {}

        for i in range(len(dataFrame)):
            if dataFrame["Dates"][i] in avgValues:
                avgValues[dataFrame["Dates"][i]] += dataFrame["Sentiment"][i]
                continue

            avgValues[dataFrame["Dates"][i]] = dataFrame["Sentiment"][i]

        for key in list(avgValues.keys()):
            avgValues[key] = round(avgValues[key], 3)

        return self.SortDict(avgValues)

    def SortDict(self, d, isDescending = False): return dict(sorted(d.items(), reverse = isDescending))
      
    

#sentEval = SentimentEval.Read_CSV("./Results/dataframe.csv")


