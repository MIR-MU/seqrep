"""
Evaluation module

This module implements the evaluation process.
It calculates the results from actual values and predictions.
It also allows visualization.
"""

import abc
from typing import List, Dict, Union
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, precision_score, recall_score

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .utils import Visualizable, visualize_labels


class Evaluator(Visualizable):
    """
    Class for evaluation of results.
    """

    def __init__(self, threshold=None):
        self.threshold = threshold

    def _transform(self, y_pred):
        """
        Transforms multi-dimensional predictions into one dimension.

        Parameters
        ----------
        y_pred : list
            Predictions of the trained model
        """

        if self.threshold is None:
            return y_pred.argmax(1)
        return (y_pred > self.threshold).argmax(1)

    @abc.abstractmethod
    def evaluate(
        self,
        y_true: Union[pd.Series, pd.DataFrame],
        y_pred: Union[pd.Series, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Calculates some metrics from y_true, y_pred.

        Parameters
        ----------
        y_true : list
            Ground truth data (original/unseen labels).

        y_pred : list
            Predictions of the trained model

        Returns
        -------
        results : dict
            Dict of calculated metric values labeled by their names.
        """
        raise NotImplementedError

    def visualize(self, y_true, y_pred) -> None:
        """
        Plot predictions and true values.

        Parameters
        ----------
        y_true : list
            Ground truth data (original/unseen labels).

        y_pred : list
            Predictions of the trained model
        """
        labels = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        return visualize_labels(
            labels=labels, title="Visualize TEST predictions and true values"
        )


class SequentialEvaluator(Evaluator):
    """
    This evaluator sequentialy triggers the entered evaluators.

    Attributes
    ----------
    evaluators_list: list
        List of Evaluators.

    visualize_only_last: bool
        Whether only the last evaluator should be used for visualization.
    """

    def __init__(
        self, evaluators_list: List[Evaluator], visualize_only_last: bool = False
    ):
        self.evaluators_list = evaluators_list
        self.visualize_only_last = visualize_only_last

    def evaluate(
        self,
        y_true: Union[pd.Series, pd.DataFrame],
        y_pred: Union[pd.Series, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Evaluates the true values and predictions using all evaluators.

        Parameters
        ----------
        y_true : list
            Ground truth data (original/unseen labels).

        y_pred : list
            Predictions of the trained model

        Returns
        -------
        results : dict
            Dict of calculated metric values labeled by their names.
        """

        results = {}
        for evaluator in self.evaluators_list:
            results.update(evaluator.evaluate(y_true, y_pred))
        return results

    def visualize(self, y_true, y_pred):
        """
        Plot the visualization either of all evaluators or only the last one.

        Parameters
        ----------
        y_true : list
            Ground truth data (original/unseen labels).

        y_pred : list
            Predictions of the trained model
        """
        if self.visualize_only_last:
            return self.evaluators_list[-1].visualize(y_true, y_pred)

        for evaluator in self.evaluators_list:
            evaluator.visualize(y_true, y_pred)


class UniversalEvaluator(Evaluator):
    """
    Evaluator which calculates provided metrics.

    Attdibutes:
    ----------
    metrics : list
        List of metrics - functions which evaluates y_true, y_pred.
    verbose : bool
        If True, the results are printed.
    """

    def __init__(self, metrics: list, verbose: bool = True):
        self.metrics = metrics
        self.verbose = verbose

    def evaluate(
        self,
        y_true: Union[pd.Series, pd.DataFrame],
        y_pred: Union[pd.Series, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        This function calculates provided metrics.
        It prints and returns their results.

        Parameters
        ----------
        y_true : list
            Ground truth data (original/unseen labels).

        y_pred : list
            Predictions of the trained model

        Returns
        -------
        results : dict
            Dict of calculated metric values labeled by their names.
        """

        results = {}
        for metric in self.metrics:
            res = metric(y_true, y_pred)
            results[metric.__name__] = res
            if self.verbose:
                res = str(res).replace("\n", "\n\t")
                print(f"{metric.__name__}:\n\t{res}")
        return results


class ClassificationEvaluator(Evaluator):
    """
    Evaluator for classification results.

    ClassificationEvaluator calculates accuracy, precision, recall and
    confusion matrix.
    """

    def evaluate(
        self,
        y_true: Union[pd.Series, pd.DataFrame],
        y_pred: Union[pd.Series, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Calculates accuracy, precision, recall and confusion matrix.

        Parameters
        ----------
        y_true : list
            Ground truth data (original/unseen labels).

        y_pred : list
            Predictions of the trained model

        Returns
        -------
        results : dict
            Dict of calculated metric values labeled by their names.
        """
        acc = accuracy_score(y_true, y_pred)
        conf_mat = confusion_matrix(y_true, y_pred)

        avg_prec = precision_score(y_true, y_pred, average=None)
        prec = sum(avg_prec[1:]) / len(avg_prec[1:])
        rec_score = recall_score(y_true, y_pred, average=None)
        rec = sum(rec_score[1:]) / len(rec_score[1:])
        print(
            conf_mat,
            "\n",
            acc * 100,
            "% accuracy\n",
            prec * 100,
            f"% precision of {len(avg_prec) -1} classes\n",
            rec * 100,
            f"% recall of {len(rec_score) -1} classes\n",
        )
        report = classification_report(y_true, y_pred)
        print(report)
        results = {
            "accuracy": acc * 100,
            "precision": prec * 100,
            "recall": rec * 100,
            "confusion matrix": conf_mat,
        }
        return results


class RegressionEvaluator(Evaluator):
    """
    Evaluator for regression results.

    RegressionEvaluator calculates Mean Absolute Error, Mean Squared Error
    (and its root) and R2 score.
    """

    def evaluate(
        self,
        y_true: Union[pd.Series, pd.DataFrame],
        y_pred: Union[pd.Series, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Calculates Mean Absolute Error, Mean Squared Error and its root,
        and R2 score.

        Parameters
        ----------
        y_true : list
            Ground truth data (original/unseen labels).

        y_pred : list
            Predictions of the trained model

        Returns
        -------
        results : dict
            Dict of calculated metric values labeled by their names.
        """
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        results = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

        print(
            f"""MAE:  {mae:>6.4f} 
MSE:  {mse:>6.4f}
RMSE: {rmse:>6.4f}
R2:   {r2:>6.4f}
"""
        )
        return results
