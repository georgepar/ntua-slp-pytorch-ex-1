from torch.utils.data import Dataset

from utils.load_data import load_semeval2017A
import utils.nlp as nlpu


class SentenceDataset(Dataset):
    def __init__(self, csv_file, word2idx, max_length=-1):
        """
        A PyTorch Dataset
        What we have to do is to implement the 2 abstract methods:

            - __len__(self): in order to let the DataLoader know the size
                of our dataset and to perform batching, shuffling and so on...
            - __getitem__(self, index): we have to return the properly
                processed data-item from our dataset with a given index
        """
        self.loaded_data = load_semeval2017A(csv_file)
        self.word2idx = word2idx
        self.max_length = max_length if max_length > 0 else self.__max_sentence_len()

    def __max_sentence_len(self):
        max_len = 0
        for _, s in self.loaded_data:
            slen = len(s)
            if slen > max_len:
                max_len = slen
        return max_len

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """
        return len(self.loaded_data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return return:
            ::
                example = [  533  3908  1387   649   0     0     0     0
                             0     0     0     0     0     0     0     0
                             0     0     0     0     0     0     0     0]
                label = 1
        """
        label, sentence = self.loaded_data[index]
        tokenized = nlpu.tokenize(sentence)
        vectorized = nlpu.vectorize(tokenized, self.word2idx, self.max_length)
        return vectorized, label, len(sentence)

