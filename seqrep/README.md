# Features
This is a list of classes of all modules with a brief description of their functionality.

## Labeling
- **Labeler**(BaseEstimator, TransformerMixin, Picklable, Visualizable):
	> _Create labels to dataset._
- **NextColorLabeler**(Labeler):
	> _NextColorLabeler applies binary labeling (0 or 1) based on the next candl._
- **NextSentimentLabeler**(Labeler):
	> _NextSentimentLabeler applies binary labeling (0 or 1) based on the direction of higher move during the next candle._
- **ClassificationLabeler**(Labeler):
	> _ClassificationLabeler applies ternary labeling according to future values._
- **RegressionLabeler**(Labeler):
	> _Find the maximum and minimum value change during selected future steps._

## Splitting
- **Splitter**(TransformerMixin, BaseEstimator, Picklable, Visualizable):
	> _Abstract class for splitting dataset._
- **TrainTestSplitter**(Splitter):
	> _Splitting to train and test taken from scikit-learn.model_selection._

## Feature_engineering
- **FeatureExtractor**(abc.ABC, BaseEstimator, TransformerMixin, Picklable):
	> _Class for implementation of feature extraction functionality._
- **FeatureSelectorExtractor**(FeatureExtractor):
	> _Select choosen features based on its names._
- **PreviousValuesExtractor**(FeatureExtractor):
	> _Add features from previous sample point._
- **TimeFeaturesExtractor**(FeatureExtractor):
	> _Add time features._
- **PandasTAExtractor**(FeatureExtractor):
	> _Add Pandas TA features._
- **TAExtractor**(FeatureExtractor):
	> _Feature extractor based on technical analysis indicators from TA library._
- **HRVExtractor**(FeatureExtractor):
	> _Add Heart Rate Variability analysis features._

## Scaling
- **Scaler**(abc.ABC, TransformerMixin, BaseEstimator, Picklable):
	> _Abstract class for scaling._
- **StandardScaler**(Scaler):
	> _Standard scaling taken from scikit-learn._
- **UniversalScaler**(Scaler):
	> _Wrapper for arbitrary scaler e.g. from scikit-learn._

## Feature_reduction
- **FeatureReductor**(BaseEstimator, TransformerMixin, Picklable, Visualizable):
	> _Class for implementation of feature reduction (selection or transformation._
- **SequentialFeatureReductor**(FeatureReductor):
	> _This reductor sequentialy triggers the entered reductors._
- **PCAReductor**(FeatureReductor):
	> _This reductor is based on Principal component analysis (PCA)._
- **FeatureSelector**(FeatureReductor):
	> _Metalass for implementation of feature selection functionality._
- **FeatureImportanceSelector**(FeatureSelector):
	> _Selects features based on feature importance._
- **UnivariateFeatureSelector**(FeatureSelector):
	> _Selects features based on univariate statistical tests._
- **RFESelector**(FeatureSelector):
	> _Selects features based on Recursive Feature Elimination._
- **VarianceSelector**(FeatureSelector):
	> _Selects features based on their variances._

## Evaluation
- **Evaluator**(Visualizable):
	> _Class for evaluation of results._
- **SequentialEvaluator**(Evaluator):
	> _This evaluator sequentialy triggers the entered evaluators._
- **UniversalEvaluator**(Evaluator):
	> _Evaluator which calculates provided metrics._
- **ClassificationEvaluator**(Evaluator):
	> _Evaluator for classification results._
- **RegressionEvaluator**(Evaluator):
	> _Evaluator for regression results._

## Pipeline_evaluation
- **PipelineEvaluator**(Picklable):
	> _PipelineEvaluator contains all modules and triggers them._

## Utils
- **Picklable**:
	> _Simple class for saving (and loading) functionality using pickle._
- **Visualizable**(abc.ABC):
	> _A simple abstract class requiring the implementation of a visualize function._