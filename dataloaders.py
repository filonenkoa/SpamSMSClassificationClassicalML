from os.path import exists
from abc import ABC, abstractmethod
import pandas as pd
from typing import List
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer


implemented_readers = ("smsspam")


class Dataloader(ABC):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_train(self):
        pass

    @abstractmethod
    def get_test(self):
        pass


class SmsSpam(Dataloader):
    """
    SMS Spam Collection v. 1, https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
    """

    def __init__(self, path: str, filter: bool, split: float = 0.8):
        assert exists(path), f"Dataset file {path} does not exist"
        self.path = path
        self.filter = filter
        self.split = split

        self.data = None
        self.train = None
        self.val = None
        self.test = None
        self.prepare_data()

    def __len__(self):
        return len(self.data)

    def prepare_data(self):
        self.read_data()
        if filter:
            self.data['clean_msg'] = self.data.message.apply(self.filter_messages)
        self.split_data()

    def read_data(self):
        self.data = pd.read_csv(self.path, encoding="latin-1")
        self.data.dropna(how="any", inplace=True, axis=1)  # dataset file contains 3 empty columns
        self.data.columns = ["label", "message"]
        # self.data["message_len"] = self.data.message.apply(len)  # can be used as a feature
        self.data["label_num"] = self.data.label.map({"ham": 0, "spam": 1})  # a numerical class value is required

    @staticmethod
    def filter_messages(mess):
        # Remove the most commonly used English words
        stop_words = stopwords.words('english')

        # Check characters to see if they are in punctuation
        nopunc = [char for char in mess if char not in string.punctuation]

        # Join the characters again to form the string.
        nopunc = ''.join(nopunc)

        # Remove any stopwords
        result = [word for word in nopunc.split() if word.lower() not in stop_words]

        # Stem words
        stemmer = EnglishStemmer()
        result = [stemmer.stem(word) for word in result]

        return " ".join(result)

    def split_data(self):
        # Get all spam and ham messages
        ham = self.data[self.data.label_num == 0]
        spam = self.data[self.data.label_num == 1]
        ham_thresh = int(len(ham) * self.split)
        spam_thresh = int(len(spam) * self.split)

        self.train = pd.concat((ham[:ham_thresh], spam[:spam_thresh]), ignore_index=True)
        self.test = pd.concat((ham[ham_thresh:], spam[spam_thresh:]), ignore_index=True)

        # Shuffle the rows
        self.train = self.train.sample(frac=1).reset_index(drop=True)

    def get_data(self):
        return self.data.clean_msg, self.data.label_num

    def get_train(self):
        return self.train.clean_msg, self.train.label_num

    def get_test(self):
        return self.test.clean_msg, self.test.label_num


if __name__ == "__main__":
    dl = SmsSpam("tests/spam_test.csv", filter=True, split=0.8)
    print(dl.get_train())
    print(dl.get_test())




