import numpy as np


def tokenize(text, lowercase=True):
    if lowercase:
        text = text.lower()
    return text.split()


def vectorize(text, word2idx, max_length):
    """
    Covert array of tokens, to array of ids, with a fixed length
    and zero padding at the end
    Args:
        text (): the wordlist
        word2idx (): dictionary of word to ids
        max_length (): the maximum length of the input sequences

    Returns: zero-padded list of ids

    """
    words = np.zeros(max_length, dtype=np.uint32)
    for i, w in enumerate(text):
        if text in word2idx:
            words[i] = word2idx[w]
        else:
            words[i] = word2idx['<unk>']
    return words

