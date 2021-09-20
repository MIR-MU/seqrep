import abc
from typing import List, Tuple

import sklearn.preprocessing

from utils import Picklable

class Scaler(abc.ABC, Picklable):

    @abc.abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the scaler
        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.
        Parameters
        ----------
        X : iterable
            Training data.
        y : iterable, default=None
            Training targets.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of used scaler.
        Returns
        -------
        self : Scaler
            This object
        """
        raise NotImplemented()

    @abc.abstractmethod
    def transform(self, X, y=None):
        """Apply scaling.
        Parameters
        ----------
        X : iterable
            Data to transform.
        Returns
        -------
        Xt : array-like of shape  (n_samples, n_transformed_features)
        """
        raise NotImplemented()

    @abc.abstractmethod
    def inverse_transform(self, X, y=None):
        """Apply inverse scaling.
        Parameters
        ----------
        Xt : array-like of shape  (n_samples, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
        """
        raise NotImplemented()


class StandardScaler(Scaler):
    
    def __init__(self):
        self.scaler = sklearn.preprocessing.StandardScaler()
    
    def fit(self, X, y=None, **fit_params):
        return self.scaler.fit(X)

    def transform(self, X, y=None):
        return self.scaler.transform(X)

    def inverse_transform(self, X, y=None):
        return self.scaler.inverse_transform(X)

class UniversalScaler(Scaler):
    
    def __init__(self, scaler):
        self.scaler = scaler
    
    def fit(self, X, y=None, **fit_params):
        return self.scaler.fit(X)

    def transform(self, X, y=None):
        return self.scaler.transform(X)

    def inverse_transform(self, X, y=None):
        return self.scaler.inverse_transform(X)