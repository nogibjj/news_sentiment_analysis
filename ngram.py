import numpy as np


def create_vocab(wordlist):  # DONE
    """create a list of unique elements from wordlist"""
    vocablist = []
    for word in wordlist:
        if word not in vocablist:
            vocablist.append(word)
            pass
    return vocablist


def keys_with_highest_value(input_dict):  # DONE
    """get the key(s) with highest value(s) from a dictionary"""
    max_value = max(input_dict.values())
    highest_keys = [key for key, value in input_dict.items() if value == max_value]
    return highest_keys


def dictionary_to_list(dict):
    result = []
    for key, value in dict.items():
        result.extend([key] * value)
    return result


def generate_n_grams(corpus, n):  # DONE i think
    """function breaking up corpus into n-grams"""
    n_gram_list = []
    if len(corpus) < n:
        return n_gram_list
    for i in range(len(corpus) - n + 1):
        n_gram = corpus[i : i + n]
        n_gram_list.append(n_gram)
        pass
    return n_gram_list


def relevant_n_grams(words, n, corpus):  # DONE I think
    """function that returns all n-grams in corpus that start with words"""
    all_grams = generate_n_grams(corpus, n)
    wordsofinterest = words[1 - n :]
    # get n-grams starting with words
    matching_n_grams = [
        n_gram for n_gram in all_grams if n_gram[: (n - 1)] == wordsofinterest
    ]
    return matching_n_grams


def create_dict(words, n, corpus):  # DONE I think
    """creates a dictionary of all n-grams beginning with words"""
    worddict = {}
    # find all instances of words in corpus
    relevant = relevant_n_grams(words, n, corpus)
    # create dictionary of the following word as key and number of occurances as value
    if relevant == []:
        return worddict
    else:
        for n_gram in relevant:
            currkey = n_gram[-1]
            if currkey in worddict.keys():
                worddict[currkey] += 1
            else:
                worddict[currkey] = 1
    return worddict


def most_probable_words(words, corpus):  # NOTDONE
    """creates a list of the equally most probable words, including backoff"""
    new_words = []
    n = len(words) + 1
    while new_words == []:
        if n == 1:
            words = []
        else:
            pass
        worddict = create_dict(words, n, corpus)
        if len(worddict) > 0:
            # append all the equally most probable words
            Keymax = keys_with_highest_value(worddict)
            new_words = Keymax
        else:
            # do stupid backoff
            n = n - 1
    return new_words


def all_possible_words(words, corpus):  # NOTDONE
    """creates a list of the words that complete all n-grams in corpus that start with words, including backoff"""
    new_words = []
    n = len(words) + 1
    while new_words == []:
        if n == 1:
            words = []
        else:
            pass
        worddict = create_dict(words, n, corpus)
        if len(worddict) > 0:
            # append all the possible endings
            new_words = dictionary_to_list(worddict)
        else:
            # do stupid backoff
            n = n - 1
    return new_words


def add_most_probable_word(finsentence, n, corpus):  # DONE I think
    """function that adds the first most probable word from list of most probable words"""
    if n == 1:
        ending = []
    elif len(finsentence) < (n - 1):
        ending = finsentence.copy()
    else:
        # break off the last n-1 words of the sentence
        ending = finsentence[(1 - n) :]
        pass
    # get most probable words
    wrdlist = most_probable_words(ending, corpus)
    # add first one
    finsentence.append(wrdlist[0])
    return finsentence


def add_random_word(finsentence, n, corpus):  # DONE I think
    """function that adds a random word from list of possible words"""
    if n == 1:
        ending = []
    elif len(finsentence) < (n - 1):
        ending = finsentence.copy()

    else:
        # break off the last n-1 words of the sentence
        ending = finsentence[(1 - n) :]
        pass
    # get most probable words
    wrdlist = all_possible_words(ending, corpus)
    # add first one
    finsentence.append(np.random.choice(wrdlist))
    return finsentence


def finish_sentence(sentence, n, corpus, randomize=False):
    """Function that takes in a list of words (sentence), and completes it either deterministically using n-grams to pick the next word based on a corpus, or randomly from the corpus vocabulary."""
    # Create a new string for our sentence that we will add to
    finalsentence = sentence.copy()
    # While the final sentence has less that 10 tokens and doesnt end with (.,!,?):
    while (len(finalsentence) < 15) and finalsentence[-1] not in (".", "!", "?"):
        if randomize == True:
            # Create Vocabulary
            vocab = create_vocab(corpus)
            # Pick a random word from vocab
            finalsentence = add_random_word(finalsentence, n, corpus)
        else:
            # Add deterministic next word
            finalsentence = add_most_probable_word(finalsentence, n, corpus)
        pass
    return finalsentence
