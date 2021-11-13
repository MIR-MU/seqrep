import abc

import numpy as np
import pandas as pd

# Finance
import pandas_ta as ta

# Health care
from hrvanalysis.extract_features import (
    get_csi_cvi_features,
    get_frequency_domain_features,
    get_geometrical_features,
    get_poincare_plot_features,
    get_sampen,
    get_time_domain_features,
)
from numpy_ext import rolling_apply
from sklearn.base import BaseEstimator
from sklearn.pipeline import TransformerMixin
from ta import (
    add_all_ta_features,
    add_momentum_ta,
    add_others_ta,
    add_trend_ta,
    add_volatility_ta,
    add_volume_ta,
)
from tqdm.auto import tqdm

from .utils import Picklable

# import talib


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
        raise NotImplementedError


class FeatureSelectorExtractor(FeatureExtractor):
    """
    Select choosen features based on its names.
    """

    def __init__(self, ordered_features: list, n: int = None):
        self.ordered_features = ordered_features
        self.n = n

    def transform(self, X, y=None):
        X.drop(
            columns=[col for col in X.columns if col not in self.ordered_features],
            inplace=True,
        )
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
        self.intervals = (
            ["minute", "hour", "weekday", "day", "weekofyear", "month", "year"]
            if intervals is None
            else intervals
        )

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
        self.indicators = (
            pd.DataFrame().ta.indicators(as_list=True)
            if indicators is None
            else indicators
        )

    def transform(self, X, y=None):
        cumulative_indicators = ["log_return", "percent_return", "trend_return"]
        for indicator in tqdm(
            self.indicators, leave=False, desc="Calculating Pandas TA indicators"
        ):
            X.ta(
                kind=indicator,
                append=True,
                cumulative=(indicator in cumulative_indicators),
            )
        return X


class TAExtractor(FeatureExtractor):
    """
    Feature extractor based on technical analysis indicators from TA library:
    https://github.com/bukosabino/ta

    Attributes:
    ----------
    all : bool
        If True, all TA features are calculated.
    volume : bool
        If True, volume features are added.
    volatility : bool
        If True, volatility features are added.
    trend : bool
        If True, trend features are added.
    momentum : bool
        If True, momentum features are added.
    others : bool
        If True, others features are added.
    fillna : bool
        If True, fill nan values.
    colprefix : str
        Prefix column names inserted.
    """

    def __init__(
        self,
        all_features: bool = False,
        volume_features: bool = False,
        volatility_features: bool = False,
        trend_features: bool = False,
        momentum_features: bool = False,
        others_features: bool = False,
        fillna: bool = False,
        colprefix: str = "",
    ):

        self.all_features = all_features
        self.volume_features = volume_features
        self.volatility_features = volatility_features
        self.trend_features = trend_features
        self.momentum_features = momentum_features
        self.others_features = others_features

        self.fillna = fillna
        self.colprefix = colprefix

    def transform(self, X, y=None) -> pd.DataFrame:
        """
        Calculates features (technical indicators).

        Parameters
        ----------
        X : iterable
            Data in OHLCV format.

        Returns
        -------
        Xt : array-like of shape  (n_samples, n_transformed_features)
            Dataset with calculated features.
        """
        if self.all_features:
            return add_all_ta_features(
                df=X,
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                fillna=self.fillna,
                colprefix=self.colprefix,
            )
        if self.volume_features:
            X = add_volume_ta(
                df=X,
                high="high",
                low="low",
                close="close",
                volume="volume",
                fillna=self.fillna,
                colprefix=self.colprefix,
            )
        if self.volatility_features:
            X = add_volatility_ta(
                df=X,
                high="high",
                low="low",
                close="close",
                fillna=self.fillna,
                colprefix=self.colprefix,
            )
        if self.trend_features:
            X = add_trend_ta(
                df=X,
                high="high",
                low="low",
                close="close",
                fillna=self.fillna,
                colprefix=self.colprefix,
            )
        if self.momentum_features:
            X = add_momentum_ta(
                df=X,
                high="high",
                low="low",
                close="close",
                volume="volume",
                fillna=self.fillna,
                colprefix=self.colprefix,
            )
        if self.others_features:
            X = add_others_ta(
                df=X, close="close", fillna=self.fillna, colprefix=self.colprefix
            )
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
    n_jobs : int, optional
        Parallel tasks count for joblib. If 1, joblib won’t be used. Default is 1.
    """

    def __init__(
        self,
        window: int = 10,
        columns: list = None,
        methods: list = [
            get_time_domain_features,
            get_geometrical_features,
            get_frequency_domain_features,
            get_csi_cvi_features,
            get_poincare_plot_features,
            get_sampen,
        ],
        n_jobs: int = 1,
    ):
        self.window = window
        self.columns = columns
        self.methods = methods
        self.n_jobs = n_jobs

    def transform(self, X, y=None):
        if self.columns is None:
            self.columns = [X.columns[-1]]

        for column in tqdm(self.columns, leave=False, desc="Calculating columns"):
            for method in tqdm(self.methods, leave=False, desc="Calculating methods"):
                features = pd.DataFrame.from_records(
                    rolling_apply(method, self.window, X[column], n_jobs=self.n_jobs)[
                        self.window - 1 :
                    ],
                    index=X.index[self.window - 1 :],
                )
                features = features.add_suffix(f"-{column}")
                X = X.join(features)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        return X
