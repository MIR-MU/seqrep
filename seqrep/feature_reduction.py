import abc

from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator

from .utils import Picklable

class FeatureReductor(abc.ABC, BaseEstimator, TransformerMixin, Picklable):
    """
    Class for implementation of feature reduction (selection or transformation)
    functionality.

    Attributes
    ----------
    number : float
        Number or proportion of features to select.
    """

    def __init__(self, number:float = 1):
        self.number = number

    @abc.abstractmethod
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
        return NotImplemented()

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


class  FeatureImportanceSelector(FeatureReductor):
    """
    Selects features based on feature importance
    
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
        self.model.fit(X, y)
        importances = self.model.feature_importances_
        features = dict(zip(X.columns, importances))
        self.sorted_features = sorted(((value, key) for (key,value) in features.items()), 
                                      reverse=True)
        return self
        
    def transform(self, X, y=None):
        """
        Transforms features according to fitted list of them.

        Returns
        -------
        X: array-like of shape  (n_samples, n_transformed_features)
            Data with selected features.
        """
        return X[[name for (_, name) in self.sorted_features][:self.number]]