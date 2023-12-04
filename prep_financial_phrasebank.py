import nltk
import pandas as pd
import random
from typing import List, Mapping, Optional, Sequence

# import gensim
import numpy as np
from numpy.typing import NDArray
from download_hf import download_parquet

FloatArray = NDArray[np.float64]

"""
This script was specifically written to prepare the financial_phrasebank dataset
    financial_phrasebank has a single sentence per row
    labels range from 0 (negative), 1(neutral), 2 (positive)
    there are 4 datasets based on % agreement between annotators
"""
import gensim.downloader as api

google_news = api.load("word2vec-google-news-300")
# google_news.save("word2vec-google-news-300.model")


# import sentiment_economy_news dataset
def import_data(split):
    """
    Import financial_phrasebank dataset
    splits available:
        sentences_50agree,
        sentences_66agree,
        sentences_75agree,
        sentences_allagree
    """
    return download_parquet("financial_phrasebank", split)


def clean_text(text):
    """Clean text."""
    # turn to lowercase
    text = text.lower()
    # split into sentences
    sentences = nltk.sent_tokenize(text)
    # remove punctuation
    sentences = [nltk.word_tokenize(sentence) for _, sentence in enumerate(sentences)]
    sentences = [
        [word for word in sentence if word.isalnum()]
        for _, sentence in enumerate(sentences)
    ]
    sentences = list(filter(None, sentences))
    return sentences


def tokenize_financial_phrasebank(df):
    """
    Tokenize sentiment economy news.
    """
    # tokenize sentences
    df["tokenized_sentences"] = df["sentence"].apply(clean_text)


def sum_token_embeddings(
    token_embeddings: Sequence[FloatArray],
) -> FloatArray:
    """Sum the token embeddings."""
    total: FloatArray = np.array(token_embeddings).sum(axis=0)
    return total


def map_labels(df):
    """Map labels to integers."""
    label_map = {0: -1, 1: 0, 2: 1}
    return df["label"].map(label_map)


def split_train_test(
    X: FloatArray, y: FloatArray, test_percent: float = 20
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Split data into training and testing sets."""
    N = len(y)
    data_idx = list(range(N))
    random.shuffle(data_idx)
    break_idx = round(test_percent / 100 * N)
    training_idx = data_idx[break_idx:]
    testing_idx = data_idx[:break_idx]
    X_train = X[training_idx, :]
    y_train = y[training_idx]
    X_test = X[testing_idx, :]
    y_test = y[testing_idx]
    return X_train, y_train, X_test, y_test


def generate_data_word2vec(df: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
    """Generate training and testing data with word2vec."""
    # load pre-trained word2vec model
    google_news = gensim.models.KeyedVectors.load("word2vec-google-news-300.model")
    X: FloatArray = np.array(
        [
            # sum the token embeddings for each sentence. If word is not in the model, return embedding of ['UNK']
            sum_token_embeddings(
                [
                    google_news[word] if word in google_news else google_news["UNK"]
                    for _, word in enumerate(sentence)
                ]
            )
            for _, sentence in enumerate(df.sentence)
        ]
    )
    y: FloatArray = np.array(map_labels(df))
    return split_train_test(X, y)


def etl(split):
    """
    Extract, transform, and load financial_phrasebank
    """
    df = import_data(split)
    tokenize_financial_phrasebank(df)
    X_train, y_train, X_test, y_test = generate_data_word2vec(df)
    return X_train, y_train, X_test, y_test


def aggregate_all_splits():
    """
    Aggregate all splits of financial_phrasebank
    """
    df = pd.DataFrame()
    for split in [
        "sentences_50agree",
        "sentences_66agree",
        "sentences_75agree",
        "sentences_allagree",
    ]:
        df = pd.concat([df, import_data(split)])
    tokenize_financial_phrasebank(df)
    X_train, y_train, X_test, y_test = generate_data_word2vec(df)
    return X_train, y_train, X_test, y_test


def run_experiment() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics

    # prepare training and testing data
    X_train, y_train, X_test, y_test = aggregate_all_splits()

    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    print("word2vec (train):", clf.score(X_train, y_train))
    print("word2vec (test):", clf.score(X_test, y_test))
