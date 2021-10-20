import abc
import pandas as pd
from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm
# Finance
import pandas_ta as ta
# Health care
from hrvanalysis.extract_features import get_time_domain_features, get_geometrical_features, get_frequency_domain_features, get_csi_cvi_features, get_poincare_plot_features, get_sampen

# import talib

from .utils import Picklable

class FeatureExtractor(abc.ABC, BaseEstimator, TransformerMixin, Picklable):
    """
    Class for implementation of feature extraction functionality.
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


class FeatureSelectorExtractor(FeatureExtractor):
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

    def transform(self, X, y=None):
        for column in X.columns:
            X.loc[:, f"{column}_shift={self.shift}"] = X[column].shift(self.shift)
        return X


class TimeFeaturesExtractor(FeatureExtractor):
    """
    Add time features.

    Attributes
    ----------
    intervals : list (string enum)
        List of datetime attributes.
    """

    def __init__(self, intervals: list = None):
        self.intervals = ['minute', 'hour', 'weekday', 'day', 'weekofyear', 'month', 'year'] if intervals is None else intervals

    def transform(self, X, y=None):
        for interval in self.intervals:
            X.loc[:, interval] = getattr(X.index, interval)
        return X


# ############################################################################
# ##########################  FINANCE FEATURES  ##############################
# ############################################################################

class PandasTAExtractor(FeatureExtractor):
    """
    Add Pandas TA features.

    Attributes
    ----------
    intervals : list (string enum)
        List of desired indicators.
    """

    def __init__(self, indicators: list = None):
        self.indicators = pd.DataFrame().ta.indicators(as_list=True) if indicators is None else indicators

    def transform(self, X, y=None):
        cumulative_indicators = ["log_return", "percent_return", "trend_return"]
        for indicator in tqdm(self.indicators, leave=False, 
                              desc="Calculating Pandas TA indicators"):
            try:
                X.ta(kind=indicator, append=True,
                     cumulative = (indicator in cumulative_indicators))
            except:
                # print("\nError with indicator:", indicator)
                pass
        return X


# ############################################################################
# ########################  HEALTH CARE FEATURES  ############################
# ############################################################################

class HRVExtractor(FeatureExtractor):
    """
    Add Heart Rate Variability analysis features.

    Attributes
    ----------
    window : int
        Size of window (number of items - i.e. rows) for calculation of statistics.
    columns : list
        Columns from which to be calculated new features.
    methods : list
        List of funkcions from hrvanalysis.extract_features to be used.
    """

    def __init__(self, window: int = 10,
                 columns: list = None,
                 methods: list = [get_time_domain_features, 
                                  get_geometrical_features,
                                  get_frequency_domain_features,
                                  get_csi_cvi_features,
                                  get_poincare_plot_features,
                                  get_sampen]
                 ):
        self.window = window
        self.columns = columns
        self.methods = methods

    def transform(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns[-1]
        statistics = []
        for i in trange(self.window, X.shape[0], 1, leave=False, desc="Calculating features"):
            start = X.index[i - self.window] 
            stop = X.index[i]
            columns_results = {}
            for column in tqdm(self.columns, leave=False, desc="Calculating columns"):
                methods_results = {}
                for method in tqdm(self.methods, leave=False, desc="Calculating methods"):
                    methods_results.update(method(X.loc[start:stop, column]))
                methods_results = {k + f"-{column}": v for k, v in methods_results.items()}
                columns_results.update(methods_results)
            statistics.append(columns_results)
        X = X.join(pd.DataFrame(statistics, index=X.index[self.window:]))
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        return X
