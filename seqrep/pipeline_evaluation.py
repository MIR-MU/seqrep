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
        labeler: Labeler = None,
        splitter: Splitter = None,
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

        if self.labeler is not None:
            self._log("Labeling data")
            self.labels = self.labeler.transform(data)
            if "labeler" in self.visualize:
                self.labeler.visualize(labels=self.labels)

        if self.splitter is not None:
            self._log("Splitting data")
            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
            ) = self.splitter.transform(X=data, y=self.labels)
            if "splitter" in self.visualize:
                self.splitter.visualize(X=[self.X_train, self.X_test])

        if self.pipeline is not None:
            self._log("Fitting pipeline")
            self.X_train = self.pipeline.fit_transform(self.X_train, self.y_train)
            self._log("Applying pipeline transformations")
            self.X_test = self.pipeline.transform(self.X_test)

        if self.dropna:
            self.X_train, self.y_train = self._drop_na(X=self.X_train, y=self.y_train)
            self.X_test, self.y_test = self._drop_na(X=self.X_test, y=self.y_test)

        if "pipeline" in self.visualize:
            visualize_data(
                X=self.X_train,
                y=self.y_train,
                downprojector=self.downprojector,
                title="Visualization of pipeline output",
            )

        if self.feature_reductor is not None:
            self._log("Applying feature reduction")
            self.feature_reductor.fit(self.X_train, self.y_train)
            self.X_train = self.feature_reductor.transform(self.X_train)
            self.X_test = self.feature_reductor.transform(self.X_test)
            if "feature_reductor" in self.visualize:
                self.feature_reductor.visualize(
                    X=self.X_train,
                    y=self.y_train,
                    downprojector=self.downprojector,
                    title="Visualization of FeatureReductor output",
                )

        if self.model is not None:
            self._log("Fitting model")
            self.model.fit(self.X_train, self.y_train)
            if "model" in self.visualize:
                self.y_pred = self.model.predict(self.X_train)
                if len(self.y_pred.shape) == 1 or self.y_pred.shape[1] == 1:
                    self.y_pred = pd.Series(self.y_pred, index=self.X_train.index)
                else:
                    self.y_pred = pd.DataFrame(self.y_pred, index=self.X_train.index)
                visualize_labels(
                    labels=pd.DataFrame(
                        {"y_true": self.y_train, "y_pred": self.y_pred}
                    ),
                    title="Visualize TRAIN predictions and true values",
                )

            self._log("Predicting")
            self.y_pred = self.model.predict(self.X_test)
            if len(self.y_pred.shape) == 1 or self.y_pred.shape[1] == 1:
                self.y_pred = pd.Series(self.y_pred, index=self.X_test.index)
            else:
                self.y_pred = pd.DataFrame(self.y_pred, index=self.X_test.index)

            if self.evaluator is not None:
                self._log("Evaluating predictions")
                result = self.evaluator.evaluate(y_true=self.y_test, y_pred=self.y_pred)
                if "evaluator" in self.visualize:
                    self.evaluator.visualize(y_true=self.y_test, y_pred=self.y_pred)
                return result
