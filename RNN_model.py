"""Sentiment analysis example, with google news word2vec and hugging face dataset"""
import nltk
import pandas as pd
import random
from typing import List, Mapping, Optional, Sequence
import numpy as np
from numpy.typing import NDArray
from download_hf import download_parquet
import torch
from torch import nn
import gensim.downloader as api

FloatArray = NDArray[np.float64]

"""
First are functions to prepare the financial_phrasebank dataset
    financial_phrasebank has a single sentence per row
    labels range from 0 (negative), 1(neutral), 2 (positive)
    there are 4 datasets based on % agreement between annotators
"""

google_news = api.load("word2vec-google-news-300")

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


def map_labels(df):
    """Map labels to integers."""
    label_map = {0: -1, 1: 0, 2: 1}
    return df["label"].map(label_map)


def generate_data_word2vec(df: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
    """Generate training and testing data with word2vec."""
    # load pre-trained word2vec model
    # google_news = gensim.models.KeyedVectors.load("word2vec-google-news-300.model")
    """Split data into training and testing sets."""
    test_percent = 30
    N = len(df)
    data_idx = list(range(N))
    random.shuffle(data_idx)
    break_idx = round(test_percent / 100 * N)
    training_idx = data_idx[break_idx:]  # Convert to NumPy array
    testing_idx = data_idx[:break_idx]
    df1 = df.iloc[training_idx]
    df2 = df.iloc[testing_idx]
    X = [
        [
            torch.tensor(
                [
                    [item]
                    for item in (
                        google_news[word] if word in google_news else google_news["UNK"]
                    ).astype("float32")
                ]
            )
            for _, word in enumerate(sentence)
        ]
        for _, sentence in enumerate(df1.sentence)
    ]
    y_true = [torch.tensor([label]) for label in np.array(map_labels(df1))]

    X_test = [
        [
            torch.tensor(
                [
                    [item]
                    for item in (
                        google_news[word] if word in google_news else google_news["UNK"]
                    ).astype("float32")
                ]
            )
            for _, word in enumerate(sentence)
        ]
        for _, sentence in enumerate(df2.sentence)
    ]
    y_test = [torch.tensor([label]) for label in np.array(map_labels(df2))]

    return X, y_true, X_test, y_test


def etl(split):
    """
    Extract, transform, and load financial_phrasebank
    """
    df = import_data(split)
    tokenize_financial_phrasebank(df)
    X, y_true, X_test, y_test = generate_data_word2vec(df)
    return X, y_true, X_test, y_test


def aggregate_all_splits():
    """
    Aggregate all splits of financial_phrasebank
    """
    df = pd.DataFrame()
    for split in [
        "sentences_75agree",
        "sentences_allagree",
    ]:
        df = pd.concat([df, import_data(split)])
    tokenize_financial_phrasebank(df)
    X, y_true, X_test, y_test = generate_data_word2vec(df)
    return X, y_true, X_test, y_test


class RNN(nn.Module):
    """Create RNN model class"""

    def __init__(self, embedding_size: int, output_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = 1
        # Layer 1 uses a linear transformation of the input and then a sigmoid activation function
        self.l1 = nn.Sequential(
            nn.Linear(self.embedding_size + self.output_size, self.hidden_size),
            nn.Sigmoid(),
        )
        # Layer 2 is just another linear transformation
        self.l2 = nn.Linear(
            self.output_size + self.hidden_size,
            self.output_size,
        )

    def forward(self, document: Sequence[torch.Tensor]) -> torch.Tensor:
        # create empty vector for output
        output = torch.zeros((self.output_size, 1), requires_grad=True)
        # run each token embedding in the document through the layers
        for token_embedding in document:
            output = self.forward_cell(token_embedding, output)
        return output

    def forward_cell(
        self, token_embedding: torch.Tensor, previous_output: torch.Tensor
    ) -> torch.Tensor:
        # puts previous output vector next to the token embedding
        concatenated = torch.cat((token_embedding, previous_output), dim=0)
        # Applies the first layer on the concatenated vector then applies the second. Transposes it just because thats how pytorch works
        result = self.l2(
            torch.cat((self.l1(concatenated.T).T, previous_output), dim=0).T
        ).T
        return result


def build_sent_model(X, y_true) -> None:
    """Train the RNN model."""
    # generate training data

    # define model
    model = RNN(300, output_size=2)
    loss_fn = torch.nn.MSELoss()

    # print initial parameters and loss
    print(
        # list(model.parameters()),
        torch.sum(
            torch.tensor(
                tuple(loss_fn(y_i, model(x_i)[0]) for x_i, y_i in zip(X, y_true))
            )
        ),
    )

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(100):  # loop over gradient descent steps
        for x_i, y_i in zip(X, y_true):  # loop over observations/"documents"
            y_pred = model(x_i)
            loss = loss_fn(y_pred[0].float(), y_i.float())  # Cast tensors to float
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # print final parameters and loss
    print(
        # list(model.parameters()),
        loss,
    )

    return model




def main():
  X, y_true, X_test, y_test = aggregate_all_splits()
  print("built training data")
  model = build_sent_model(X, y_true)
  print("finished training model")
  
  X_new = X
  y_new = y_true
  
  print("making predictions")
  predictions = [model(x_i) for x_i in X_new]
  
  predictions = [pred[0].item() for pred in predictions]
  
  thresholds = np.arange(0.01, 0.91, 0.05).tolist()
  
  columns = ["upper threshold", "lower threshold", "train accuracy"]
  results_df1 = pd.DataFrame(columns=columns)
  
  for iii, thresh in enumerate(thresholds):
      for ii, thresh2 in enumerate(thresholds):
          threshup = thresh
          threshdown = -thresh2
          correct = 0
          wrong = 0
          accuracy = 0
          for i, pred in enumerate(predictions):
              true = y_new[i][0]
              if pred > threshup:
                  guess = 1
              if pred < threshdown:
                  guess = -1
              else:
                  guess = 0
  
              if guess == true:
                  correct += 1
              else:
                  wrong += 1
          total = correct + wrong
          accuracy = correct / total
  
          results_df1.loc[len(results_df1.index)] = {
              "upper threshold": threshup,
              "lower threshold": threshdown,
              "train accuracy": accuracy,
          }
  
  results_df1 = results_df1.sort_values(by="train accuracy", ascending=False)
  
  top_row1 = results_df1.iloc[0]
  print(top_row1)
  
  X_new = X_test
  y_new = y_test
  
  predictions = [model(x_i) for x_i in X_new]
  
  predictions = [pred[0].item() for pred in predictions]
  
  thresholds = np.arange(0.01, 0.91, 0.05).tolist()
  
  columns = ["upper threshold", "lower threshold", "test accuracy"]
  results_df = pd.DataFrame(columns=columns)
  
  for iii, thresh in enumerate(thresholds):
      for ii, thresh2 in enumerate(thresholds):
          threshup = thresh
          threshdown = -thresh2
          correct = 0
          wrong = 0
          accuracy = 0
          for i, pred in enumerate(predictions):
              true = y_new[i][0]
              if pred > threshup:
                  guess = 1
              if pred < threshdown:
                  guess = -1
              else:
                  guess = 0
  
              if guess == true:
                  correct += 1
              else:
                  wrong += 1
          total = correct + wrong
          accuracy = correct / total
  
          results_df.loc[len(results_df.index)] = {
              "upper threshold": threshup,
              "lower threshold": threshdown,
              "test accuracy": accuracy,
          }
  
  results_df = results_df.sort_values(by="test accuracy", ascending=False)
  
  top_row = results_df.iloc[0]
  print(top_row)
