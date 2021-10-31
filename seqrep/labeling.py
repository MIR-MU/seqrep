import abc
import pandas as pd

from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator
from .utils import Picklable


class Labeler(abc.ABC, BaseEstimator, TransformerMixin, Picklable):
    """
    Create labels to dataset.
    """

    def fit(self, X, y=None, **fit_params):
        return self

    @abc.abstractmethod
    def transform(self, X, y=None):
        raise NotImplementedError


class NextColorLabeler(Labeler):
    """
    NextColorLabeler applies binary labeling (0 or 1) based on the next candle
    if it is bullish or bearish.

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
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=["open", "high", "low", "close", "volume"])
        labels = (
            (X[self.column_end].shift(-1) - X[self.column_start]).shift(-1) > 0
        ).astype(int)
        return labels.fillna(0)


class RegressionLabeler(Labeler):
    """
    Find the maximum and minimum value change during selected future steps.

    Parameters
    ----------
    duration: int
        Maximal length for reaching label value
    """

    def __init__(self, duration: int = 1):
        self.duration = duration

    def transform(self, X, y=None):
        labels = pd.DataFrame(index=X.index)
        labels["positive_label"] = (
            X["high"].shift(-self.duration).rolling(self.duration).max()
        )
        labels["negative_label"] = (
            X["low"].shift(-self.duration).rolling(self.duration).min()
        )
        for i in range(self.duration - 1):
            labels.loc[labels.index[i], ("positive_label",)] = X["close"][
                i : i + self.duration - 1
            ].max()
            labels.loc[labels.index[i], ("negative_label",)] = X["close"][
                i : i + self.duration - 1
            ].min()
        labels["positive_label"] -= X["close"]  # to do or not ?
        labels["negative_label"] -= X["close"]  # to do or not ?
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
