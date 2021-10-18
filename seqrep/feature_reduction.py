import abc
from typing import Optional, Union

from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, f_classif

from .utils import Picklable

class FeatureReductor(abc.ABC, BaseEstimator, TransformerMixin, Picklable):
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
        raise NotImplemented()


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
        selected_columns = [name for (_, name) in self.sorted_features][:self.number]
        return X[selected_columns]


class  FeatureImportanceSelector(FeatureSelector):
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
        features = dict(zip(X.columns, importances))
        self.sorted_features = sorted(((value, key) for (key,value) in features.items()), 
                                      reverse=True)
        return self


class  UnivariateFeatureSelector(FeatureSelector):
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
        features = dict(zip(X.columns, selector.scores_))
        self.sorted_features = sorted(((value, key) for (key,value) in features.items()),
                                      reverse=True)
        return self