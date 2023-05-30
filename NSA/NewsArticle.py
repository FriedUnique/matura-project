from enum import Enum
from datetime import datetime

class ArticleSentiment:
    class Sentiment(Enum):
        Negative = -1
        Neutral = 0
        Positive = 1

    def __init__(self, rawValue: float, sentiment: Sentiment):
        self.raw = rawValue
        self.sentiment = sentiment


class Article:
    """Represents a news Article"""
    class Section(Enum):
        """Article Category"""
        All = "all"
        World = "world"
        Business = "business"
        Legal = "legal"
        Tech = "technology"

    class DateRange(Enum):
        """Article Date Range Selection"""
        Any = "any_time"
        Day = "past_24_hours"
        Week = "past_week"
        Month = "past_month"
        Year = "past_year"

    def __init__(self, date: datetime, title: str, body: str, sentiment: ArticleSentiment):
        self.date = date
        self.title = title
        self.body = body
        self.sentiment = sentiment


