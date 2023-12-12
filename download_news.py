import requests
import datetime
import pandas as pd

API_KEY = "<YOUR_API_KEY>"

today = datetime.datetime.now().strftime("%Y-%m-%d")


def days_back(days):
    return (pd.to_datetime(today) - datetime.timedelta(days=days)).strftime("%Y-%m-%d")


def list_of_days(from_date=None):
    today = pd.to_datetime("today")
    from_date = pd.to_datetime(from_date)
    days = (today - from_date).days + 1
    workdays = pd.bdate_range(end=today, periods=days).strftime("%Y-%m-%d").to_list()
    return workdays


# using Everything endpoint
def everything_news(date, keywords=None):
    """
    Returns a list of articles from the day specified
    q
        Keywords or phrases to search for in the article title and body.

        Advanced search is supported here:

        Surround phrases with quotes (") for exact match.
        Prepend words or phrases that must appear with a + symbol. Eg: +bitcoin
        Prepend words that must not appear with a - symbol. Eg: -bitcoin
        Alternatively you can use the AND / OR / NOT keywords, and optionally group these with parenthesis. Eg: crypto AND (ethereum OR litecoin) NOT bitcoin.
        The complete value for q must be URL-encoded. Max length: 500 chars
    """

    url = (
        "https://newsapi.org/v2/everything?"
        "q={}&"
        "seachIn=description&"
        "from={}&"
        "to={}&"
        "sortBy=popularity&"
        "language=en&"
        "apiKey={}"
    ).format(keywords, date, date, API_KEY)
    response = requests.get(url)
    return response.json()


def create_news_df(news):
    """
    Extracts description and source from news json and returns a dataframe
    """
    news_df = pd.DataFrame(columns=["date", "source", "text"])
    for article in news.get("articles"):
        news_df = pd.concat(
            [
                news_df,
                pd.DataFrame.from_dict(
                    {
                        "date": [
                            pd.to_datetime(article.get("publishedAt")).strftime(
                                "%Y-%m-%d"
                            )
                        ],
                        "source": [article.get("source").get("name")],
                        "text": [article.get("description")],
                    }
                ),
            ],
            ignore_index=True,
        )
    return news_df


def df_from_days(days_list, keywords=None):
    """
    Returns a dataframe with news from the days in the list
    """
    news_df = pd.DataFrame(columns=["date", "source", "text"])
    for day in days_list:
        news = everything_news(day, keywords)
        if news.get("status") == "ok":
            news_df = pd.concat([news_df, create_news_df(news)], ignore_index=True)
        else:
            pass
    return news_df


# using top-headlines endpoint
def top_news(category=None, keywords=None):
    """
    Returns a list of articles using top headlines YESTERDAY
    This doesn't allow to filter for language => we filter by country=us
    category
        The category you want to get headlines for.
        Possible options: business, entertainment, general, health,
        science, sports, technology.
        Note: you can't mix this param with the sources param.
    """
    url = (
        "https://newsapi.org/v2/top-headlines?"
        "country=us&"
        "category={}&"
        "q={}&"
        "apiKey={}"
    ).format(category, keywords, API_KEY)

    response = requests.get(url)
    return response.json()
