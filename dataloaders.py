from os.path import exists
from abc import ABC, abstractmethod
import pandas as pd
from typing import List
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer

# def get_implemented_readers():
#     """
#     Add all the implemented datasets here
#     :return:
#     """
#     datasets = dict(
#         "smsspam"=SmsSpam.path
#     )
#     return datasets


class Dataloader(ABC):

    @abstractmethod
    def __len__(self):
        pass


class SmsSpam(Dataloader):
    """
    SMS Spam Collection v. 1, https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
    """

    def __init__(self, path: str, filter: bool, split: List[float] = (0.7, 0.2, 0.1)):
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
        self.data["message_len"] = self.data.message.apply(len)  # can be used as a feature
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
        ham_thresh = (int(len(ham) * self.split[0]), int(len(ham) * (self.split[0] + self.split[1])))
        spam_thresh = (int(len(spam) * self.split[0]), int(len(spam) * (self.split[0] + self.split[1])))
        self.train = pd.concat((ham[0:ham_thresh[0]], spam[0:spam_thresh[0]]), ignore_index=True)
        self.val = pd.concat((ham[ham_thresh[0]:ham_thresh[1]], spam[spam_thresh[0]:spam_thresh[1]]), ignore_index=True)
        self.test = pd.concat((ham[ham_thresh[1]:], spam[spam_thresh[1]:]), ignore_index=True)

        # Shuffle the rows
        self.train = self.train.sample(frac=1).reset_index(drop=True)

    def get_train(self):
        return self.train

    def get_val(self):
        return self.val

    def get_test(self):
        return self.test


if __name__ == "__main__":
    dl = SmsSpam("tests/spam_test.csv", filter=True, split=[0.7, 0.2, 0.1])
    print(dl.get_train())
    print(dl.get_val())
    print(dl.get_test())




