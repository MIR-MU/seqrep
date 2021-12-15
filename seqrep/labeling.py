"""
Labeling module

With this module you can create labels for the data.
"""

import abc

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import TransformerMixin

from .utils import Picklable, Visualizable, visualize_labels


class Labeler(BaseEstimator, TransformerMixin, Picklable, Visualizable):
    """
    Create labels to dataset.
    """

    def fit(self, X, y=None, **fit_params):
        """
        Returns self (it doesn't do anything with the data).
        
        Parameters
        ----------
        X : iterable
            Data to transform.
        
        y : iterable, default=None
            Training targets.
        """
        return self

    @abc.abstractmethod
    def transform(self, X, y=None):
        """
        Calculates the labels and returns them.
        
        Parameters
        ----------
        X : iterable
            Data to transform.
        
        y : iterable, default=None
            Training targets.        
        """
        raise NotImplementedError

    def visualize(self, labels, X=None, mode: str = "lines") -> None:
        """
        Plot labels.
        """
        return visualize_labels(labels=labels, mode=mode)


class NextColorLabeler(Labeler):
    """
    NextColorLabeler applies binary labeling (0 or 1) based on the next candle
    color if it is bullish or bearish.

    Parameters
    ----------
    column_start: str
        Column name of the first considered value of the future candle.

    column_end: str
        Column name of the second considered value of the future candle.
    """

    def __init__(self, column_start="open", column_end="close"):
        self.column_start = column_start
        self.column_end = column_end

    def transform(self, X, y=None):
        """
        Calculates the binary labels from next sample point.

        Parameters
        ----------
        X : iterable
            Data to transform.
        
        y : iterable, default=None
            Training targets.

        Returns
        -------
        labels: pd.DataFrame or pd.Series
            Calculated labels.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=["open", "high", "low", "close", "volume"])
        labels = ((X[self.column_end] - X[self.column_start]).shift(-1) > 0).astype(int)
        return labels


class NextSentimentLabeler(Labeler):
    """
    NextSentimentLabeler applies binary labeling (0 or 1) based on the direction of higher move during the next candle.

    Parameters
    ----------
    positive: str
        Column of the move in the positive direction.

    negative: str
        Column of the move in the negative direction.

    base: str
        Column of the reference value.
    """

    def __init__(
        self, positive: str = "high", negative: str = "low", base: str = "open"
    ):
        self.positive = positive
        self.negative = negative
        self.base = base

    def transform(self, X, y=None):
        """
        Calculates the binary labels from next sample point.

        Parameters
        ----------
        X : iterable
            Data to transform.
        
        y : iterable, default=None
            Training targets.
        """
        labels = (
            (
                (X[self.positive] - X[self.base]).shift(-1)
                > (X[self.base] - X[self.negative]).shift(-1)
            )
        ).astype(int)
        return labels


class ClassificationLabeler(Labeler):
    """
    ClassificationLabeler applies ternary labeling according to future values.

    Parameters
    ----------
    duration: int
        Maximal length for reaching label value,
        i.e number of future datapoints

    pip_size: float
        Size of one pip in units of currency (usually 0.0001)

    target_profit: float
        Size of the target profit in pips

    stop_loss: float
        Size of the stop loss in pips
    """

    def __init__(
        self,
        duration: int = 1,
        target_profit: float = 20,
        stop_loss: float = 15,
        pip_size: float = 0.0001,
    ):
        self.duration = duration
        self.pip_size = pip_size
        self.target_profit = target_profit
        self.stop_loss = stop_loss

    def transform(self, X, y=None):
        """
        Calculates the labels according to target_profit and stop_loss values.

        Parameters
        ----------
        X : iterable
            Data to transform.
        
        y : iterable, default=None
            Training targets.
        """
        labels = RegressionLabeler(duration=self.duration).transform(X)
        labels["label"] = 0
        labels["label"] = labels["label"].mask(
            (labels["positive_label"] >= self.target_profit * self.pip_size)
            & (-labels["negative_label"] <= self.stop_loss * self.pip_size),
            1,
        )
        labels["label"] = labels["label"].mask(
            (-labels["negative_label"] >= self.target_profit * self.pip_size)
            & (labels["positive_label"] <= self.stop_loss * self.pip_size),
            2,
        )
        return labels.drop(columns=["positive_label", "negative_label"])


# ############################################################################
# #############################  REGRESSION  #################################
# ############################################################################


class RegressionLabeler(Labeler):
    """
    Find the maximum and minimum value change during selected future steps.

    Parameters
    ----------
    duration: int
        Maximal length for reaching label value
    
    positive: str
        Column of the move in the positive direction.

    negative: str
        Column of the move in the negative direction.

    base: str
        Column of the reference value.
    """

    def __init__(self, duration: int = 1, positive: str = "high", negative: str = "low", base: str = "close"):
        self.duration = duration
        self.positive = positive
        self.negative = negative
        self.base = base

    def transform(self, X, y=None):
        """
        Finds maximal and minimal values in n future samples,
        where n is the duration.

        Parameters
        ----------
        X : iterable
            Data to transform.
        
        y : iterable, default=None
            Training targets.
        """
        labels = pd.DataFrame(index=X.index)
        labels["positive_label"] = (
            X[self.positive].shift(-self.duration).rolling(self.duration).max()
        )
        labels["negative_label"] = (
            X[self.negative].shift(-self.duration).rolling(self.duration).min()
        )
        for i in range(self.duration - 1):
            labels.loc[labels.index[i], ("positive_label",)] = X[self.positive][
                i : i + self.duration - 1
            ].max()
            labels.loc[labels.index[i], ("negative_label",)] = X[self.negative][
                i : i + self.duration - 1
            ].min()
        labels["positive_label"] -= X[self.base]
        labels["negative_label"] -= X[self.base]
        return labels