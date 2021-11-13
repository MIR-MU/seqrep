import abc
from typing import List

import pandas as pd
import plotly.graph_objects as go
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import TransformerMixin

from .utils import Picklable, Visualizable


class Splitter(TransformerMixin, BaseEstimator, Picklable, Visualizable):
    """
    Abstract class for splitting dataset.
    """

    def fit(self, X, y=None, **fit_params):
        return self

    @abc.abstractmethod
    def transform(self, X, y=None):
        """
        Split X and y.
        """
        raise NotImplementedError

    def visualize(self, X: List, y=None):
        """
        Visualize train and test subsets.
        """
        X_train, X_test = X[:2]
        fig = go.Figure()
        fig.update_layout(title="Visualization of train-test split")
        fig.update_xaxes(title_text="time")
        fig.update_yaxes(title_text=X_train.columns[0])
        fig.add_trace(
            go.Scatter(
                x=X_train.index,
                y=X_train.iloc[:, 0],
                name="train",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=X_test.index,
                y=X_test.iloc[:, 0],
                name="test",
                mode="lines",
            )
        )
        fig.show()


class TrainTestSplitter(Splitter):
    """
    Splitting to train and test taken from scikit-learn.model_selection.
    """

    def __init__(
        self,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=False,
        stratify=None,
    ):
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify

    def transform(self, X, y, verbose=False):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=self.stratify,
        )
        return X_train, X_test, y_train, y_test
