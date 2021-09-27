import datetime

from .labeling import Labeler
from .splitting import Splitter
from .evaluation import Evaluator


class PipelineEvaluator():
    """
    PipelineEvaluator contains all modules and triggers them.
    """

    def __init__(self, labeler: Labeler, splitter: Splitter, pipeline, 
                 evaluator: Evaluator, verbose=True):
        self.labeler = labeler
        self.splitter = splitter
        self.pipeline = pipeline
        self.evaluator = evaluator
        self.verbose = verbose

    def _log(self, text):
        """
        Print actual time and provided text if verobse is True.

        Parameters
        ----------
        text: string
            Comment added to printed time.
        """

        if self.verbose:
            print(datetime.datetime.now().time(), text)
    
    def run(self, data):
        """
        Run each module on provided data.

        Parameters
        ----------
            data: array-like
                Data to evaluate the pipeline on.

        Returns
        -------
        y_pred : list
            List of predicted values on the test-data subset.
        """

        self._log("Labeling data")
        labels = self.labeler.transform(data)
        self._log("Splitting data")
        X_train, X_test, y_train, y_test = self.splitter.transform(X=data, y=labels)
        self._log("Fitting pipeline")
        self.pipeline.fit(X_train, y_train)
        self._log("Predicting")
        y_pred = self.pipeline.predict(X_test)
        self._log("Evaluating predictions")
        self.evaluator.evaluate(y_true=y_test, y_pred=y_pred)
        return y_pred
    