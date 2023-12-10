import nltk

nltk.download("gutenberg")
nltk.download("punkt")

from ngram import finish_sentence
import random as rand


def generate_observation(seed, corpus):
    """Test Markov text generator."""
    rand.seed(seed)

    first_word = rand.choice(corpus)

    words = finish_sentence(
        [first_word],
        3,
        corpus,
        randomize=False,
    )

    string = " ".join(words)

    return string


if __name__ == "__main__":
    corpus = nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())
    for i in range(10):
        print(f"observation {i + 1}: {generate_observation(i, corpus)}")
