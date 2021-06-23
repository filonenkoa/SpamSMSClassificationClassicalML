"""
Machine learning text classification solvers
"""
from abc import ABC, abstractmethod
from time import time
import logging
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

from dataloaders import Dataloader


implemented_solvers = ("logistic_regression")


class Solver:

    def __init__(self, dataloader: Dataloader, debug: bool):
        self.dataloader = dataloader
        self.debug = debug

    @abstractmethod
    def train(self):
        pass


class LogisticRegressionSolver(Solver):

    def __init__(self, dataloader: Dataloader, debug: bool = True):
        super().__init__(dataloader, debug)
        self.solver = None

    def print(self, msg: str):
        if self.debug:
            logging.info(msg)

    def train(self):
        x_train_dtm, y_train, x_test_dtm, y_test = self.prepare_data()

        solver = LogisticRegression(solver='liblinear')  # liblinear works better with small datasets
        self.print("Fitting the model")
        start = time()
        solver.fit(x_train_dtm, y_train)
        end = time()
        self.print(f"Done fitting in {1000 * (end-start):.2f} ms")

        self.print("Testing")
        y_pred_class = solver.predict(x_test_dtm)
        accuracy = metrics.accuracy_score(y_test, y_pred_class)
        self.print(f"Accuracy: {accuracy:.3f}")

        confusion_matrix = metrics.confusion_matrix(y_test, y_pred_class)
        self.print(f"Confusion matrix\n{confusion_matrix}")

        y_pred_prob = solver.predict_proba(x_test_dtm)[:, 1]
        roc_auc_score = metrics.roc_auc_score(y_test, y_pred_prob)
        self.print(f"ROC AUC score {roc_auc_score:.3f}")

        self.solver = solver

    def prepare_data(self):
        vect = CountVectorizer(stop_words='english', max_df=0.5)

        self.print("Learning data vocabulary")
        vect.fit(self.dataloader.get_data()[0])
        x_train, y_train = self.dataloader.get_train()
        x_test, y_test = self.dataloader.get_test()

        x_train_dtm = vect.transform(x_train)  # document-term matrix
        x_test_dtm = vect.transform(x_test)
        self.print("Done learning data vocabulary")

        self.print("Transforming a count matrix to a normalized tf-idf representation")
        tfidf_transformer = TfidfTransformer()
        tfidf_transformer.fit(x_train_dtm)
        tfidf_transformer.transform(x_train_dtm)
        self.print("Done getting the tf-idf representation")

        return x_train_dtm, y_train, x_test_dtm, y_test


def get_solver(name: str) -> Solver:
    if name.lower() == "logistic_regression":
        return LogisticRegressionSolver
    else:
        raise Exception(f"Solver {name} is not implemented")




