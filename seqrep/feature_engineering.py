import abc

from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator
from .utils import Picklable

class FeatureExtractor(abc.ABC, BaseEstimator, TransformerMixin, Picklable):

    def fit(self, X, y=None, **fit_params):
        return self

    @abc.abstractmethod
    def transform(self, X, y=None):
        raise NotImplemented()


class FeatureSelector(FeatureExtractor):
    
    def __init__(self, ordered_features: list, n: int = None):
        self.ordered_features = ordered_features
        self.n = n
        # ic(self.ordered_features)
    
    def transform(self, X, y=None):
        # ic(self.ordered_features)
        # X = X.loc[:, self.ordered_features[:self.n]] #.copy()
        # return X.loc[:, self.ordered_features[:self.n]]
        X.drop(columns=[col for col in X.columns if col not in self.ordered_features], inplace=True)
        return X[self.ordered_features]


class PreviousValuesExtractor(FeatureExtractor):

    def __init__(self, shift: int = None):
        self.shift = 1 if shift is None else shift

    def transform(self, X, y=None):            # TODO: solve nan
        for column in X.columns:
            X.loc[:, f"{column}_shift={self.shift}"] = X[column].shift(self.shift).fillna(X[column][0])
        return X