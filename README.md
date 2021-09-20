# SeqRep

*Scientific framework for representation in sequential data*

## Description

This package aims to simplify the workflow of **evaluation of machine learning models**. It is primarily focused on sequential data. It helps with:

- labeling data,
- splitting data,
- **feature extraction** (or selection),
- running pipeline,
- evaluation of results.

The framework is designed for easy customization and extension of its functionality.

## Installation

```bash
python -m pip install git+https://github.com/MIR-MU/seqrep
```

## Usage

It is simple to use this package. After the import, you need to do three steps:

1. Create your *pipeline* (which you want to evaluate);
2. Create *PipelineEvaluator* (according to how you want to evaluate);
3. Run the evaluation.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from seqrep.feature_engineering import PreviousValuesExtractor
from seqrep.labeling import NextColorLabeler
from seqrep.splitting import TrainTestSplitter
from seqrep.evaluation import ClassificationEvaluator
from seqrep.pipeline_evaluation import PipelineEvaluator

# 1. step
pipe = Pipeline([('fext_prev', PreviousValuesExtractor()),
                 ('scale', MinMaxScaler()),
                 ('svc', SVC()),
                 ])
# 2. step
pipe_eval = PipelineEvaluator(labeler = NextColorLabeler(),
                              splitter = TrainTestSplitter(),
                              pipeline = pipe,
                              evaluator = ClassificationEvaluator(),
                              )
# 3. step
result = pipe_eval.run(data=data)
```



## License

MIT

## Acknowledgement

Thanks for the huge support to my supervisor [Michal Stefanik](https://github.com/stefanik12)! Gratitude also belongs to all members of the [MIR-MU](https://github.com/MIR-MU/) group. Finally, thanks go to the Faculty of Informatics of Masaryk University for supporting this project as a dean's project.