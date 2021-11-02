import abc
from typing import List
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report, precision_score, recall_score

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .utils import Visualizable


class Evaluator(Visualizable):
    """
    Class for evaluation of results.
    """

    def __init__(self, threshold=None):
        self.threshold = threshold

    def _transform(self, y_pred):
        if self.threshold is None:
            return y_pred.argmax(1)
        return (y_pred > self.threshold).argmax(1)

    @abc.abstractmethod
    def evaluate(self, y_true, y_pred):
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
        results : list
            List of calculated metric values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def visualize(self, y_true, y_pred):
        raise NotImplementedError


class SequentialEvaluator(Evaluator):
    """
    This evaluator sequentialy triggers the entered evaluators.

    Attributes
    ----------
    evaluators_list: list
        List of Evaluators.
    """

    def __init__(
        self, evaluators_list: List[Evaluator], visualize_only_last: bool = False
    ):
        self.evaluators_list = evaluators_list
        self.visualize_only_last = visualize_only_last

    def evaluate(self, y_true, y_pred):
        """
        Evaluates the true values and predictions.
        """
        results = []
        for evaluator in self.evaluators_list:
            results += evaluator.evaluate(y_true, y_pred)
        return results

    def visualize(self, y_true, y_pred):
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

    def evaluate(self, y_true, y_pred):
        """
        This function calculates provided metrics.
        It prints and returns their results.
        """
        results = []
        for metric in self.metrics:
            res = metric(y_true, y_pred)
            results.append(res)
            if self.verbose:
                res = str(res).replace("\n", "\n\t")
                print(f"{metric.__name__}:\n\t{res}")
        return results

    def visualize(self, y_true, y_pred):
        pass  # TODO


class ClassificationEvaluator(Evaluator):
    """
    Evaluator for classification results.

    ClassificationEvaluator calculates accuracy, precision, recall and
    confusion metrix.
    """

    def evaluate(self, y_true, y_pred):
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
        results = [
            acc * 100,
            prec * 100,
            rec * 100,
        ]
        return results

    def visualize(self, y_true, y_pred):
        pass  # TODO


class RegressionEvaluator(Evaluator):
    """
    Evaluator for regression results.

    RegressionEvaluator calculates Mean Absolute Error, Mean Squared Error
    (and its root) and R2 score.
    """

    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        results = [mae, mse, rmse, r2]

        print(
            f"""MAE:  {mae:>6.4f} 
MSE:  {mse:>6.4f}
RMSE: {rmse:>6.4f}
R2:   {r2:>6.4f}
"""
        )
        return results

    def visualize(self, y_true, y_pred):
        pass  # TODO
