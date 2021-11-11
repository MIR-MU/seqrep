import abc
from typing import Optional, Union, List

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from .utils import Picklable, Visualizable, visualize_data


class FeatureReductor(BaseEstimator, TransformerMixin, Picklable, Visualizable):
    """
    Class for implementation of feature reduction (selection or transformation)
    functionality.
    """

    def fit(self, X, y=None, **fit_params):
        """
        Fits the selected method.

        Parameters
        ----------
        X: iterable
            Features for selection.
        y: iterable
            Labels for selection.

        Returns
        -------
        self: object
            Fitted reductor.
        """
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
        raise NotImplementedError

    def visualize(
        self,
        X,
        y,
        downprojector=None,
        title: str = "Visualization of FeatureReductor output",
    ) -> None:
        return visualize_data(X=X, y=y, downprojector=downprojector, title=title)


class SequentialFeatureReductor(FeatureReductor):
    """
    This reductor sequentialy triggers the entered reductors.

    Attributes
    ----------
    reductors_list: list
        List of FeatureReductors.
    """

    def __init__(self, reductors_list: List[FeatureReductor]):
        self.reductors_list = reductors_list

    def fit(self, X, y, **fit_params):
        """
        Fits all reductors from the list.

        Parameters
        ----------
        X: iterable
            Features for selection.
        y: iterable
            Labels for selection.

        Returns
        -------
        self: object
            Fitted reductor.
        """
        for reductor in self.reductors_list[:-1]:
            X = reductor.fit_transform(X, y)
        self.reductors_list[-1].fit(X, y)
        return self

    def transform(self, X, y=None):
        """
        Gradually applies transformations of all reductors from the list.

        Parameters
        ----------
        X : iterable
            Data to transform.

        Returns
        -------
        Xt : array-like of shape  (n_samples, n_transformed_features)
        """
        for reductor in self.reductors_list:
            X = reductor.transform(X)
        return X


class PCAReductor(FeatureReductor):
    """
    This reductor is based on Principal component analysis (PCA):
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Attributes
    ----------
    n_components : int, float, default=None
        Number of components to keep.

    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        See the sklearn documentation for details.
    """

    def __init__(
        self, n_components: Optional[Union[int, float]] = None, svd_solver="auto"
    ):
        self.n_components = n_components
        self.svd_solver = svd_solver

    def fit(self, X, y, **fit_params):
        """
        Fit the PCA object with X.

        Parameters
        ----------
        X: iterable
            Features for selection.
        y: iterable
            Labels for selection.

        Returns
        -------
        self: object
            Fitted reductor.
        """
        self.pca = PCA(
            n_components=self.n_components,
            svd_solver=self.svd_solver,
        ).fit(X)
        return self

    def transform(self, X, y=None):
        """
        Apply the dimensionality reduction on X.

        Parameters
        ----------
        X : iterable
            Data to transform.

        Returns
        -------
        Xt : array-like of shape  (n_samples, n_transformed_features)
        """
        return pd.DataFrame(self.pca.transform(X), index=X.index)


# ############################################################################
# ########################### FEATURE SELECTION ##############################
# ############################################################################


class FeatureSelector(FeatureReductor):
    """
    Metalass for implementation of feature selection functionality.

    Attributes
    ----------
    number : float
        Number or proportion of features to select.
    """

    def __init__(self, number: Optional[Union[int, float]] = 1):
        self.number = number

    def fit(self, X, y=None, **fit_params):
        """
        Fits the selected method.

        Parameters
        ----------
        X: iterable
            Features for selection.
        y: iterable
            Labels for selection.

        Returns
        -------
        self: object
            Fitted reductor.
        """
        if self.number <= 1:
            self.number = int(X.shape[1] * self.number)
        return self

    def transform(self, X, y=None):
        """
        Transforms features according to the fitted list of them.

        Returns
        -------
        X: array-like of shape  (n_samples, n_transformed_features)
            Data with selected features.
        """
        selected_columns = [name for (_, name) in self.sorted_features][: self.number]
        return X[selected_columns]


class FeatureImportanceSelector(FeatureSelector):
    """
    Selects features based on feature importance.

    Attributes
    ----------
    model:
        Model determining the performance.
        It has to have `feature_importances_` attribute.

    sorted_features: list
        List of pairs (value, feature_name) where the value specifies
        the importance of the feature.
    """

    def __init__(self, model, number):
        self.model = model
        super(FeatureImportanceSelector, self).__init__(number)

    def fit(self, X, y, **fit_params):
        """
        Train model and save the list of features with their importances.

        Returns
        -------
        self: object
            Fitted reductor.
        """
        super(FeatureImportanceSelector, self).fit(X)

        self.model.fit(X, y)
        importances = self.model.feature_importances_
        features = dict(zip(importances, X.columns))
        self.sorted_features = sorted(features.items(), reverse=True)
        return self


class UnivariateFeatureSelector(FeatureSelector):
    """
    Selects features based on univariate statistical tests.

    This selector is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest.

    Attributes
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores.
        Default is f_classif. The default function only
        works with classification tasks.

    sorted_features: list
        List of pairs (value, feature_name) where the value specifies
        the score of the feature.
    """

    def __init__(self, number, score_func=f_classif):
        self.score_func = score_func
        super(UnivariateFeatureSelector, self).__init__(number)

    def fit(self, X, y, **fit_params):
        """
        Calculates the univariate scores and save them with the feature names
        in a list.

        Returns
        -------
        self: object
            Fitted selector.
        """
        super(UnivariateFeatureSelector, self).fit(X)

        selector = SelectKBest(score_func=self.score_func)
        selector.fit(X, y)
        features = dict(zip(selector.scores_, X.columns))
        self.sorted_features = sorted(features.items(), reverse=True)
        return self


class RFESelector(FeatureSelector):
    """
    Selects features based on Recursive Feature Elimination.

    This selector is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

    Attributes
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance
        (e.g. `coef_`, `feature_importances_`).

    sorted_features : list
        List of pairs (value, feature_name) where the value specifies
        the score of the feature.
    """

    def __init__(
        self,
        number,
        estimator=LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=280),
    ):
        self.estimator = estimator
        super(RFESelector, self).__init__(number)

    def fit(self, X, y, **fit_params):
        """
        Calculates the feature importances and save them with the feature names
        in a list.

        Returns
        -------
        self: object
            Fitted selector.
        """
        super(RFESelector, self).fit(X)

        selector = RFE(self.estimator, 1)
        selector.fit(X, y)

        values = 1 - (selector.ranking_ / max(selector.ranking_))
        features = dict(zip(values, X.columns))
        self.sorted_features = sorted(features.items(), reverse=True)
        return self


class VarianceSelector(FeatureSelector):
    """
    Selects features based on their variances.

    This selector is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html.

    Attributes
    ----------
    sorted_features: list
        List of pairs (value, feature_name) where the value specifies
        the score of the feature.
    """

    def __init__(self, number, score_func=f_classif):
        self.score_func = score_func
        super(VarianceSelector, self).__init__(number)

    def fit(self, X, y, **fit_params):
        """
        Calculates the variances and save them with the feature names in a list.

        Returns
        -------
        self: object
            Fitted selector.
        """
        super(VarianceSelector, self).fit(X)

        selector = VarianceThreshold()
        selector.fit(X, y)
        features = dict(zip(selector.variances_, X.columns))
        self.sorted_features = sorted(features.items(), reverse=True)
        return self
