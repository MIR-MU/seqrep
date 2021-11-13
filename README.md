# SeqRep
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
[![Check Markdown links](https://github.com/MIR-MU/seqrep/actions/workflows/action.yml/badge.svg)](https://github.com/MIR-MU/seqrep/actions/workflows/action.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MIR-MU/seqrep/blob/main/examples/SimpleClassificationExample.ipynb)
[![CodeFactor](https://www.codefactor.io/repository/github/mir-mu/seqrep/badge/main)](https://www.codefactor.io/repository/github/mir-mu/seqrep/overview/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

*Scientific framework for representation in sequential data*

## Description

This package aims to simplify the workflow of **evaluation of machine learning models**. It is primarily focused on sequential data. It helps with:

- labeling data,
- splitting data,
- **feature extraction**,
- feature reduction (i.e. selection or transformation),
- running pipeline,
- evaluation of results.

The framework is designed for easy customization and extension of its functionality.

## Installation

```bash
python -m pip install git+https://github.com/MIR-MU/seqrep
```
## Features
See the `README` in the [seqrep folder](seqrep).

## Usage

It is simple to use this package. After the import, you need to do three steps:

1. Create your *pipeline* (which you want to evaluate);
2. Create *PipelineEvaluator* (according to how you want to evaluate);
3. Run the evaluation.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from seqrep.feature_engineering import PreviousValuesExtractor, TimeFeaturesExtractor
from seqrep.labeling import NextColorLabeler
from seqrep.splitting import TrainTestSplitter
from seqrep.scaling import UniversalScaler
from seqrep.evaluation import ClassificationEvaluator
from seqrep.pipeline_evaluation import PipelineEvaluator

# 1. step
pipe = Pipeline([('fext_prev', PreviousValuesExtractor()),
                 ('fext_time', TimeFeaturesExtractor()),
                 ('scale_u', UniversalScaler(scaler=MinMaxScaler())),
                 ])

# 2. step
pipe_eval = PipelineEvaluator(labeler = NextColorLabeler(),
                              splitter = TrainTestSplitter(),
                              pipeline = pipe,
                              model = SVC(),
                              evaluator = ClassificationEvaluator(),
                              )
# 3. step
result = pipe_eval.run(data=data)
```
See the [examples folder](examples) for more details.

## License

[MIT](LICENSE)

## Acknowledgement

Thanks for the huge support to my supervisor [Michal Stefanik](https://github.com/stefanik12)! Gratitude also belongs to all members of the [MIR-MU](https://github.com/MIR-MU/) group. Finally, thanks go to the Faculty of Informatics of Masaryk University for supporting this project as a dean's project.
