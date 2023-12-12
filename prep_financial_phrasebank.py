import nltk
import pandas as pd
import random
from typing import List, Mapping, Optional, Sequence
from afinn import Afinn
import gensim.downloader as api
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


google_news = api.load("word2vec-google-news-300")
# google_news.save("word2vec-google-news-300.model")

afin = Afinn()


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

nltk.download("wordnet")
nltk.download("stopwords")


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


def sum_vader_scores(sentence):
    sent_score = np.sum(np.array([afin.score(word) for word in sentence]))
    return sent_score


def generate_X_vader(df):
    X = np.array([sum_vader_scores(sentence) for sentence in df["tokenized_sentences"]])
    return X.reshape(-1, 1)


def generate_y_vader(df):
    y = np.array(df.label).reshape(-1, 1)
    return y


def etl(split):
    """
    Extract, transform, and load financial_phrasebank
    """
    df = import_data(split)
    df = tokenize_financial_phrasebank(df)
    X_train, y_train, X_test, y_test = generate_data_word2vec(df)
    return X_train, y_train, X_test, y_test


def import_all_splits():
    """
    Import all splits of financial_phrasebank
    """
    df = pd.DataFrame()
    for split in [
        "sentences_50agree",
        "sentences_66agree",
        "sentences_75agree",
        "sentences_allagree",
    ]:
        df = pd.concat([df, import_data(split)])
    return df


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


def run_experiment(X_train, y_train, X_test, y_test) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics

    # prepare training and testing data
    # X_train, y_train, X_test, y_test = aggregate_all_splits()

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


def RandomForest_experiment(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics

    # prepare training and testing data
    # X_train, y_train, X_test, y_test = aggregate_all_splits()

    rfc = RandomForestClassifier(random_state=0, max_depth=10, n_estimators=120).fit(
        X_train, y_train
    )
    # print("word2vec (train):", rfc.score(X_train, y_train))
    # print("word2vec (test):", rfc.score(X_test, y_test))

    return rfc


def GradientBoost_experiment(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn import metrics

    # prepare training and testing data
    # X_train, y_train, X_test, y_test = aggregate_all_splits()

    gbc = GradientBoostingClassifier(
        random_state=0, max_depth=4, n_estimators=30, learning_rate=0.3
    ).fit(X_train, y_train)
    # print("word2vec (train):", gbc.score(X_train, y_train))
    # print("word2vec (test):", gbc.score(X_test, y_test))

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
    X_train, y_train, X_test, y_test = aggregate_all_splits()

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
            self.fc1 = nn.Linear(300, 400)  # First fully connected layer
            self.fc2 = nn.Linear(400, 200)  # Second fully connected layer
            self.fc3 = nn.Linear(200, 3)  # Output layer

        def forward(self, x):
            x = F.relu(self.fc1(x))  # Activation function for the first layer
            x = torch.sigmoid(
                self.fc2(x)
            )  # Sigmoid activation function for the second layer
            x = self.fc3(x)  # No activation function for the output layer
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
            # running_loss += loss.item()
            # if i % 100 == 99:  # print every 100 mini-batches
            #     print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))

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
    print("word2vec accuracy: ", accuracy)


def attention_experiment():
    """
    NOT WORKING
    """
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader

    class Attention(nn.Module):
        def __init__(self, hidden_size):
            super(Attention, self).__init__()
            self.hidden_size = hidden_size
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.rand(hidden_size))

        def forward(self, hidden, encoder_outputs):
            timestep = encoder_outputs.size(0)
            h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
            encoder_outputs = encoder_outputs.transpose(0, 1)
            attn_energies = self.score(h, encoder_outputs)
            return nn.functional.softmax(attn_energies, dim=1).unsqueeze(1)

        def score(self, hidden, encoder_outputs):
            energy = nn.functional.relu(
                self.attn(torch.cat([hidden, encoder_outputs], 2))
            )
            energy = energy.transpose(1, 2)
            v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
            energy = torch.bmm(v, energy)
            return energy.squeeze(1)

    class Net(nn.Module):
        def __init__(
            self, input_size=300, hidden_size=400, num_layers=2, num_classes=3
        ):
            super(Net, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
            self.attention = Attention(hidden_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, h0)
            attn_weights = self.attention(out, out)
            out = torch.bmm(attn_weights, out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    def run_experiment():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train, y_train, X_test, y_test = aggregate_all_splits()

        X_train = torch.tensor(X_train).float().to(device)
        y_train = torch.tensor(y_train).long().to(device)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model = Net(input_size=300, hidden_size=400, num_layers=2, num_classes=3)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{10} Loss: {loss.item()}")

        X_test = torch.tensor(X_test).float().to(device)
        y_test = torch.tensor(y_test).long().to(device)

        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            "Accuracy of the network on the test data: %d %%" % (100 * correct / total)
        )

    run_experiment()
