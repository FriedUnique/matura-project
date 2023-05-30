
import yfinance


class MarketSentiment:
    def __init__(self):
        self.SNP = yfinance.download("^GSPC", period="max", interval="1d")
        
        options_data = yfinance.Ticker("^GSPC")
        options_chain = options_data.option_chain(date=options_data.options[0])  # Get options chain for the nearest expiration date


    def MarketMomentum(self):
        MA_125 = self.SNP["Close"].rolling(125).mean()
        print(self.SNP["Close"][-1] > MA_125[-1])

    def PutCallRatio(self):
        puts = options_chain.puts
        calls = options_chain.calls

        # Calculate the 5-day Put/Call ratio
        put_call_ratio = puts.openInterest.tail(5).sum() / calls.openInterest.tail(5).sum()
        print(put_call_ratio)

sent = MarketSentiment().PutCallRatio()