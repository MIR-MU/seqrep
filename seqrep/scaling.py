"""
Scaling module

This module implements wrappers for the scaling function for use them in sklearn.Pipeline.
"""

import abc
from typing import List, Tuple

import pandas as pd
from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator
import sklearn.preprocessing

from .utils import Picklable


class Scaler(abc.ABC, TransformerMixin, BaseEstimator, Picklable):
    """
    Abstract class for scaling.
    """

    @abc.abstractmethod
    def fit(self, X, y=None, **fit_params):
        """
        Fit the scaler.

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
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, X, y=None):
        """
        Apply scaling.

        Parameters
        ----------
        X : iterable
            Data to transform.
        
        y : iterable, default=None
            Training targets.

        Returns
        -------
        Xt : array-like of shape  (n_samples, n_transformed_features)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_transform(self, X, y=None):
        """
        Apply inverse scaling.

        Parameters
        ----------
        Xt : array-like of shape  (n_samples, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
        """
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        """
        Calls the fit function and then the transform.

        Parameters
        ----------
        X : iterable
            Data to transform.
        
        y : iterable, default=None
            Training targets.

        Returns
        -------
        Xt : array-like of shape  (n_samples, n_transformed_features)
        """
        self.fit(X, y)
        return self.transform(X, y)


class StandardScaler(Scaler):
    """
    Standard scaling taken from scikit-learn:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html.
    """

    def __init__(self):
        self.scaler = sklearn.preprocessing.StandardScaler()

    def fit(self, X, y=None, **fit_params):
        """
        Calls the fit function of the StandardScaler.

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
        return self.scaler.fit(X)

    def transform(self, X, y=None):
        """
        Calls the transform function of the StandardScaler.

        Parameters
        ----------
        X : iterable
            Data to transform.
        
        y : iterable, default=None
            Training targets.

        Returns
        -------
        X : array-like of shape  (n_samples, n_transformed_features)
        """
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)

    def inverse_transform(self, X, y=None):
        """
        Calls the inverse_transform function of the StandardScaler.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        
        y : iterable, default=None
            Training targets.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
        """
        return pd.DataFrame(
            self.scaler.inverse_transform(X), columns=X.columns, index=X.index
        )


class UniversalScaler(Scaler):
    """
    Wrapper for arbitrary scaler e.g. from scikit-learn.

    Attributes
    ----------
    scaler : callable
        Scaler function e.g. from scikit-learn.
    """

    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X, y=None, **fit_params):
        """
        Calls the fit function of the selected scaler.

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
        Xt = X
        return self.scaler.fit(Xt)

    def transform(self, X, y=None):
        """
        Calls the transform function of the selected scaler.

        Parameters
        ----------
        X : iterable
            Data to transform.
        
        y : iterable, default=None
            Training targets.

        Returns
        -------
        Xt : array-like of shape  (n_samples, n_transformed_features)
        """
        Xt = X
        return pd.DataFrame(self.scaler.transform(Xt), columns=X.columns, index=X.index)

    def inverse_transform(self, X, y=None):
        """
        Calls the inverse_transform function of the selected scaler.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        
        y : iterable, default=None
            Training targets.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
        """
        return pd.DataFrame(
            self.scaler.inverse_transform(X), columns=X.columns, index=X.index
        )