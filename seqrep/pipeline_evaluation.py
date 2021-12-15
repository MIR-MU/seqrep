"""
Pipeline Evaluation module

This module runs all the steps used and allows you to visualize them.
"""

import datetime
from typing import List, Tuple, Union

import pandas as pd
from sklearn.pipeline import Pipeline

from .evaluation import Evaluator
from .feature_reduction import FeatureReductor
from .labeling import Labeler
from .splitting import Splitter
from .utils import Picklable, visualize_data, visualize_labels


class PipelineEvaluator(Picklable):
    """
    PipelineEvaluator contains all modules and triggers them.
    """

    def __init__(
        self,
        labeler: Labeler,
        splitter: Splitter,
        pipeline: Pipeline = None,
        feature_reductor: FeatureReductor = None,
        model=None,
        evaluator: Evaluator = None,
        dropna: bool = True,
        downprojector=None,
        visualize: Union[bool, List[str]] = False,
        verbose: bool = True,
    ):
        self.labeler = labeler
        self.splitter = splitter
        self.pipeline = pipeline
        self.feature_reductor = feature_reductor
        self.model = model
        self.evaluator = evaluator
        self.dropna = dropna
        self.downprojector = downprojector
        self.visualize = visualize
        self.verbose = verbose

        if isinstance(self.visualize, bool):
            if self.visualize:
                self.visualize = [
                    "labeler",
                    "splitter",
                    "pipeline",
                    "feature_reductor",
                    "model",
                    "evaluator",
                ]
            else:
                self.visualize = []

    def _log(self, text) -> None:
        """
        Print actual time and provided text if verobse is True.

        Parameters
        ----------
        text: string
            Comment added to printed time.
        """

        if self.verbose:
            print(datetime.datetime.now().time().strftime("%H:%M:%S.%f")[:-3], text)

    def _drop_na(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Drop rows with NaN values from begining.

        Returns
        -------
        X, y : tupple (pd.DataFrame, pd.Series)
            X as data (with features) and y as labels.
        """

        original_shape = X.shape
        X.dropna(axis=1, thresh=int(X.shape[0] * 0.9), inplace=True)
        cut_number = X.isna().sum().max()
        X = X.iloc[cut_number:, :]
        if X.isna().sum().sum() > 0:
            X = X.dropna(axis=0)
        y = y.loc[X.index]
        self._log(
            f"\tOriginal shape:\t\t{original_shape}; \n\t\tshape after removing NaNs: {X.shape}."
        )
        return X, y

    def run(self, data):
        """
        Run each module on provided data.

        Parameters
        ----------
        data : array-like
            Data to evaluate the pipeline on.

        Returns
        -------
        result : dict
            Dict of calculated metric values labeled by their names.
        """

        self._log("Labeling data")
        labels = self.labeler.transform(data)
        if "labeler" in self.visualize:
            self.labeler.visualize(labels=labels)

        self._log("Splitting data")
        X_train, X_test, y_train, y_test = self.splitter.transform(X=data, y=labels)
        if "splitter" in self.visualize:
            self.splitter.visualize(X=[X_train, X_test])

        if self.pipeline is not None:
            self._log("Fitting pipeline")
            X_train = self.pipeline.fit_transform(X_train, y_train)
            self._log("Applying pipeline transformations")
            X_test = self.pipeline.transform(X_test)

        if self.dropna:
            X_train, y_train = self._drop_na(X=X_train, y=y_train)
            X_test, y_test = self._drop_na(X=X_test, y=y_test)

        if "pipeline" in self.visualize:
            visualize_data(
                X=X_train,
                y=y_train,
                downprojector=self.downprojector,
                title="Visualization of pipeline output",
            )

        if self.feature_reductor is not None:
            self._log("Applying feature reduction")
            self.feature_reductor.fit(X_train, y_train)
            X_train = self.feature_reductor.transform(X_train)
            X_test = self.feature_reductor.transform(X_test)
            if "feature_reductor" in self.visualize:
                self.feature_reductor.visualize(
                    X=X_train,
                    y=y_train,
                    downprojector=self.downprojector,
                    title="Visualization of FeatureReductor output",
                )

        if self.model is not None:
            self._log("Fitting model")
            self.model.fit(X_train, y_train)
            if "model" in self.visualize:
                y_pred = self.model.predict(X_train)
                if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
                    y_pred = pd.Series(y_pred, index=X_train.index)
                else:
                    y_pred = pd.DataFrame(y_pred, index=X_train.index)
                visualize_labels(
                    labels=pd.DataFrame({"y_true": y_train, "y_pred": y_pred}),
                    title="Visualize TRAIN predictions and true values",
                )

            self._log("Predicting")
            y_pred = self.model.predict(X_test)
            if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
                y_pred = pd.Series(y_pred, index=X_test.index)
            else:
                y_pred = pd.DataFrame(y_pred, index=X_test.index)

            if self.evaluator is not None:
                self._log("Evaluating predictions")
                result = self.evaluator.evaluate(y_true=y_test, y_pred=y_pred)
                if "evaluator" in self.visualize:
                    self.evaluator.visualize(y_true=y_test, y_pred=y_pred)
                return result

        return X_train, X_test, y_train, y_test