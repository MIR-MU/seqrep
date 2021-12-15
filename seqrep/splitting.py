"""
Splitting module

This module provides splitting data to subsets for training and testing.
"""

import abc
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator
import plotly.graph_objects as go

from .utils import Picklable, Visualizable


class Splitter(TransformerMixin, BaseEstimator, Picklable, Visualizable):
    """
    Abstract class for splitting dataset.
    """

    def fit(self, X, y=None, **fit_params):
        """
        Returns self (it doesn't do anything with the data).
        """
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
    Splitting to train and test taken from scikit-learn.model_selection
    (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
    
    Attributes
    ----------
    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    shuffle : bool, default=False
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.
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
        """
        Split pandas DataFrame into train and test subsets.
        """
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