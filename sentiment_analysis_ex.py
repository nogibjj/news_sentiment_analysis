"""Sentiment analysis example, with "not"."""
import random
from typing import List, Mapping, Optional, Sequence
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

FloatArray = NDArray[np.float64]


class RNN(nn.Module):
    def __init__(self, embedding_size: int, output_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = 1
        self.l1 = nn.Sequential(
            nn.Linear(self.embedding_size + self.output_size, self.hidden_size),
            nn.Sigmoid(),
        )
        self.l2 = nn.Linear(
            self.output_size + self.hidden_size,
            self.output_size,
        )

    def forward(self, document: Sequence[torch.Tensor]) -> torch.Tensor:
        output = torch.zeros((self.output_size, 1), requires_grad=True)
        for token_embedding in document:
            output = self.forward_cell(token_embedding, output)
        return output

    def forward_cell(
        self, token_embedding: torch.Tensor, previous_output: torch.Tensor
    ) -> torch.Tensor:
        concatenated = torch.cat((token_embedding, previous_output), dim=0)
        result = self.l2(
            torch.cat((self.l1(concatenated.T).T, previous_output), dim=0).T
        ).T
        return result


def generate_observation(length: int) -> tuple[list[str], float]:
    document = [random.choice(("good", "bad", "uh", "not")) for _ in range(length)]
    sentiment = 0.0
    invert = False
    for token in document:
        if (token == "good" and not invert) or (token == "bad" and invert):
            sentiment += 1
        elif (token == "bad" and not invert) or (token == "good" and invert):
            sentiment += -1
        invert = token == "not"
    return document, sentiment


def onehot(
    vocabulary_map: Mapping[Optional[str], int], token: Optional[str]
) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary_map), 1))
    idx = vocabulary_map.get(token, len(vocabulary_map) - 1)
    embedding[idx, 0] = 1
    return embedding


def rnn_example() -> None:
    """Demonstrate a simple RNN."""
    # generate training data
    observation_count = 1000
    max_length = 10
    lengths = np.round(np.random.rand(observation_count) * max_length)
    observations = [generate_observation(round(length)) for length in lengths]

    vocabulary = sorted(
        set(token for sentence, _ in observations for token in sentence)
    )
    vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}
    X = [
        [
            torch.tensor(onehot(vocabulary_map, token).astype("float32"))
            for token in sentence
        ]
        for sentence, _ in observations
    ]
    y_true = [torch.tensor([label]) for _, label in observations]

    # define model
    model = RNN(len(vocabulary), output_size=2)
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
            loss = loss_fn(y_pred[0], y_i)
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # print final parameters, loss and accuracy
    print(
        list(model.parameters()),
        loss,
    )


if __name__ == "__main__":
    rnn_example()
