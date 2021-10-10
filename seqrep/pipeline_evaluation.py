import datetime

from .labeling import Labeler
from .splitting import Splitter
from sklearn.pipeline import Pipeline
from .feature_reduction import FeatureReductor
from .evaluation import Evaluator

class PipelineEvaluator():
    """
    PipelineEvaluator contains all modules and triggers them.
    """

    def __init__(self, labeler: Labeler, splitter: Splitter, 
                 pipeline: Pipeline = None, 
                 feature_reductor: FeatureReductor = None, model = None, 
                 evaluator: Evaluator = None, 
                 verbose: bool=True):
        self.labeler = labeler
        self.splitter = splitter
        self.pipeline = pipeline
        self.feature_reductor = feature_reductor
        self.model = model
        self.evaluator = evaluator
        self.verbose = verbose

    def _log(self, text) -> None:
        """
        Print actual time and provided text if verobse is True.

        Parameters
        ----------
        text: string
            Comment added to printed time.
        """

        if self.verbose:
            print(datetime.datetime.now().time().strftime('%H:%M:%S.%f')[:-3],
                  text)
    
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

        if self.pipeline is not None:
            self._log("Fitting pipeline")
            X_train = self.pipeline.fit_transform(X_train, y_train)
            self._log("Applying pipeline transformations")
            X_test = self.pipeline.transform(X_test)


        if self.feature_reductor is not None:
            self._log("Applying feature reduction")
            self.feature_reductor.fit(X_train, y_train)            
            X_train = self.feature_reductor.transform(X_train)
            X_test = self.feature_reductor.transform(X_test)

        if self.model is not None:
            self._log("Fitting model")
            self.model.fit(X_train, y_train)

            self._log("Predicting")
            y_pred = self.model.predict(X_test)

            if self.evaluator is not None:
                self._log("Evaluating predictions")
                self.evaluator.evaluate(y_true=y_test, y_pred=y_pred)
                return y_pred