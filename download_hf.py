import requests
import pandas as pd


datasets = {
    # this dataset only has positive and negative labels
    "sentiment_economy_news": {
        "author": "MoritzLaurer",
        "train": "https://huggingface.co/api/datasets/MoritzLaurer/sentiment_economy_news/parquet/default/train",
        "test": "https://huggingface.co/api/datasets/MoritzLaurer/sentiment_economy_news/parquet/default/test",
    },
    # this dataset only contains headlines. There are 5 labels
    "stock_news_sentiment": {
        "author": "ic-fspml",
        "train": "https://huggingface.co/api/datasets/ic-fspml/stock_news_sentiment/parquet/default/train",
        "test": "https://huggingface.co/api/datasets/ic-fspml/stock_news_sentiment/parquet/default/test",
    },
    # labels here are 0, 1, 2
    "ml_news_sentiment": {
        "author": "sara-nabhani",
        "train": "https://huggingface.co/api/datasets/sara-nabhani/ML-news-sentiment/parquet/default/train",
        "test": "https://huggingface.co/api/datasets/sara-nabhani/ML-news-sentiment/parquet/default/test",
    },
    # financial phrasebank datasets divided by % agreement between annotators:
    "financial_phrasebank": {
        "sentences_50agree": "https://huggingface.co/api/datasets/financial_phrasebank/parquet/sentences_50agree/train",
        "sentences_66agree": "https://huggingface.co/api/datasets/financial_phrasebank/parquet/sentences_66agree/train",
        "sentences_75agree": "https://huggingface.co/api/datasets/financial_phrasebank/parquet/sentences_75agree/train",
        "sentences_allagree": "https://huggingface.co/api/datasets/financial_phrasebank/parquet/sentences_allagree/train",
    },
}


def get_parquet_url(dataset, split="train"):
    url = datasets[dataset][split]

    def query():
        response = requests.get(url)
        return response.json()

    data = query()
    return data[0]


def download_parquet(dataset, split="train"):
    url = get_parquet_url(dataset, split)
    df = pd.read_parquet(url)
    return df
