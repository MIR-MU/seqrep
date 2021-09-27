import abc

from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator
from .utils import Picklable

class FeatureExtractor(abc.ABC, BaseEstimator, TransformerMixin, Picklable):
    """
    Class for implementation of feature extraction and feature selection 
    functionality.
    """

    def fit(self, X, y=None, **fit_params):
        return self

    @abc.abstractmethod
    def transform(self, X, y=None):
        """
        Extract or select features.
        
        Parameters
        ----------
        X : iterable
            Data to transform.
        
        Returns
        -------
        Xt : array-like of shape  (n_samples, n_transformed_features)
        """
        raise NotImplemented()


class FeatureSelector(FeatureExtractor):
    """
    Select choosen features based on its names.
    """

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
    """
    Add features from previous sample point.
    """

    def __init__(self, shift: int = None):
        self.shift = 1 if shift is None else shift

    def transform(self, X, y=None):            # TODO: solve nan
        for column in X.columns:
            X.loc[:, f"{column}_shift={self.shift}"] = X[column].shift(self.shift).fillna(X[column][0])
        return X