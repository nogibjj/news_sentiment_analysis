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


from nltk.corpus import wordnet


def Negation(sentence):
    """
    Input: Tokenized sentence (List of words)
    Output: Tokenized sentence with negation handled (List of words)
    """
    temp = int(0)
    for i in range(len(sentence)):
        if sentence[i - 1] in ["not", "n't"]:
            antonyms = []
            for syn in wordnet.synsets(sentence[i]):
                syns = wordnet.synsets(sentence[i])
                w1 = syns[0].name()
                temp = 0
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
                max_dissimilarity = 0
                for ant in antonyms:
                    syns = wordnet.synsets(ant)
                    w2 = syns[0].name()
                    syns = wordnet.synsets(sentence[i])
                    w1 = syns[0].name()
                    word1 = wordnet.synset(w1)
                    word2 = wordnet.synset(w2)
                    if isinstance(word1.wup_similarity(word2), float) or isinstance(
                        word1.wup_similarity(word2), int
                    ):
                        temp = 1 - word1.wup_similarity(word2)
                    if temp > max_dissimilarity:
                        max_dissimilarity = temp
                        antonym_max = ant
                        sentence[i] = antonym_max
                        sentence[i - 1] = ""
    while "" in sentence:
        sentence.remove("")
    return sentence


def clean_text(text):
    """Clean text incorporating Negation handling and stopwords."""
    # turn to lowercase
    text = text.lower()
    # word tokenization
    text = nltk.word_tokenize(text)
    # negation handling
    text = Negation(text)
    # remove punctuation
    text = [word for word in text if word.isalnum()]
    # remove stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    text = [word for word in text if word not in stopwords]
    if text == "":
        pass
    else:
        return " ".join(text)


# def clean_text(text):
#     """Clean text."""
#     # turn to lowercase
#     text = text.lower()
#     # split into sentences
#     sentences = nltk.sent_tokenize(text)
#     # remove punctuation
#     sentences = [nltk.word_tokenize(sentence) for _, sentence in enumerate(sentences)]
#     sentences = [
#         [word for word in sentence if word.isalnum()]
#         for _, sentence in enumerate(sentences)
#     ]
#     sentences = list(filter(None, sentences))
#     return sentences


def tokenize_financial_phrasebank(df):
    """
    Tokenize sentiment economy news.
    """
    # tokenize sentences
    df["tokenized_sentences"] = df["sentence"].apply(clean_text)
    df = df.loc[df.tokenized_sentences != ""]
    return df


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
    # google_news = gensim.models.KeyedVectors.load("word2vec-google-news-300.model")
    X: FloatArray = np.array(
        [
            # sum the token embeddings for each sentence. If word is not in the model, return embedding of ['UNK']
            sum_token_embeddings(
                [
                    google_news[word] if word in google_news else google_news["UNK"]
                    for _, word in enumerate(sentence)
                ]
            )
            for _, sentence in enumerate(df.tokenized_sentences)
        ]
    )
    # labels = [-1, 0, 1] seems to be causing an error
    # y: FloatArray = np.array(map_labels(df))
    y: FloatArray = np.array(df.label)
    return split_train_test(X, y)


def generate_observation_word2vec(sentence):
    X: FloatArray = np.array(
        [
            sum_token_embeddings(
                [
                    google_news[word] if word in google_news else google_news["UNK"]
                    for _, word in enumerate(sentence)
                ]
            )
        ]
    )
    return X


def etl(split):
    """
    Extract, transform, and load financial_phrasebank
    """
    df = import_data(split)
    df = tokenize_financial_phrasebank(df)
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
    df = tokenize_financial_phrasebank(df)
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

    return clf


def experiment_gridSearchCV():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["newton-cg", "lbfgs", "saga"],
    }
    # prepare training and testing data
    X_train, y_train, X_test, y_test = aggregate_all_splits()
    clf = LogisticRegression(random_state=0, max_iter=1000)

    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("Best Score: ", grid_search.best_score_)
    print("Best Params: ", grid_search.best_params_)


def RandomForest_experiment():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics

    # prepare training and testing data
    X_train, y_train, X_test, y_test = aggregate_all_splits()

    rfc = RandomForestClassifier(random_state=0, max_depth=10, n_estimators=120).fit(
        X_train, y_train
    )
    print("word2vec (train):", rfc.score(X_train, y_train))
    print("word2vec (test):", rfc.score(X_test, y_test))

    return rfc


def GradientBoost_experiment():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn import metrics

    # prepare training and testing data
    X_train, y_train, X_test, y_test = aggregate_all_splits()

    gbc = GradientBoostingClassifier(
        random_state=0, max_depth=4, n_estimators=30, learning_rate=0.3
    ).fit(X_train, y_train)
    print("word2vec (train):", gbc.score(X_train, y_train))
    print("word2vec (test):", gbc.score(X_test, y_test))

    return gbc


def MLP_experiment():
    from sklearn.neural_network import MLPClassifier
    from sklearn import metrics

    # prepare training and testing data
    X_train, y_train, X_test, y_test = aggregate_all_splits()

    mlp = MLPClassifier(random_state=0, max_iter=1000).fit(X_train, y_train)
    print("word2vec (train):", mlp.score(X_train, y_train))
    print("word2vec (test):", mlp.score(X_test, y_test))


def RNN_experiment_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from sklearn import metrics

    # prepare training and testing data
    # X_train, y_train, X_test, y_test = aggregate_all_splits()
    X_train, y_train, X_test, y_test = etl("sentences_allagree")

    # convert to torch tensors
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    # create dataset
    class FinancialPhraseBankDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # create dataloader
    train_dataset = FinancialPhraseBankDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # define model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(300, 100)
            self.fc2 = nn.Linear(100, 3)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    net = Net()

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))

    print("Finished Training")
    # After training, generate predictions on test data
    X_test = X_test.float()
    outputs_test = net(X_test)
    _, predicted = torch.max(outputs_test, 1)

    # Convert tensors to numpy arrays for comparison with sklearn metrics
    y_test_np = y_test.numpy()
    predicted_np = predicted.numpy()

    # Now you can use sklearn's metrics to compare y_test_np and predicted_np
    # For example, to calculate accuracy:
    accuracy = metrics.accuracy_score(y_test_np, predicted_np)
    print("Accuracy: ", accuracy)
