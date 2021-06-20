from os.path import exists
from abc import ABC, abstractmethod
import pandas as pd

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
    def get_num_classes(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


class SmsSpam(Dataloader):
    """
    SMS Spam Collection v. 1, https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
    """

    def __init__(self, path: str, filter: bool):
        assert exists(path), f""
        self.path = path
        if filter:
            self.filter_dataset()


    def filter_dataset(self):



