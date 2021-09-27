import abc
from sklearn.model_selection import train_test_split
from .utils import Picklable

class Splitter(abc.ABC, Picklable):
    """
    Abstract class for splitting dataset.
    """
    
    def fit(self, X, y=None, **fit_params):
        return self

    @abc.abstractmethod
    def transform(self, X, y=None):
        raise NotImplemented()


class TrainTestSplitter(Splitter):
    """
    Splitting to train and test taken from scikit-learn.model_selection.
    """

    def __init__(self, test_size=None, train_size=None, random_state=None, shuffle=False, stratify=None):
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify

    def transform(self, X, y, verbose=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, train_size=self.train_size, 
                                                            random_state=self.random_state, shuffle=self.shuffle, stratify=self.stratify)
        return X_train, X_test, y_train, y_test


# class TrainValTestSplitter(Splitter): # TODO: implement