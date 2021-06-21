"""
Machine learning text classification solvers
"""
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from time import time

from dataloaders import Dataloader


implemented_solvers = ("logistic_regression")

class Solver(ABC):

    @abstractmethod
    def __init__(self, dataloader: Dataloader, debug: bool):
        self.dataloader = dataloader
        self.debug = debug

    @abstractmethod
    def train(self, iterations: int):
        pass


class LogisticRegressionSolver(Solver):

    def __init__(self, dataloader: Dataloader, debug: bool = True):
        super().__init__(dataloader, debug)

    def print(self, msg: str):
        if self.debug:
            print(msg)

    def train(self, iterations: int):
        vect = CountVectorizer()

        self.print("Learning data vocabulary")
        vect.fit(self.dataloader.get_data()[0])
        x_train, y_train = self.dataloader.get_train()
        x_test, y_test = self.dataloader.get_test()

        x_train_dtm = vect.transform(x_train)  # document-term matrix
        x_test_dtm = vect.transform(x_test)
        self.print("Done learning data vocabulary")

        solver = LogisticRegression()
        self.print("Fitting the model")
        start = time()
        solver.fit(x_train_dtm, y_train)
        end = time()
        self.print(f"Done fitting in {end-start} seconds")





