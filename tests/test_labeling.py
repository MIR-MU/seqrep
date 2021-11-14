import pandas as pd

from seqrep.labeling import *


def test_NextColorLabeler():
    labeler = NextColorLabeler(column_start="open", column_end="close")
    data = pd.DataFrame({"open": [0, 2, 1], "close": [0, 4, 0]})
    true_labels = pd.Series([1, 0, 0])  # 4>2 -> "1", 0<1 -> "0"
    labels = labeler.transform(data)
    assert labels[0] == true_labels[0]
    assert labels[1] == true_labels[1]


def test_RegressionLabeler():
    labeler = RegressionLabeler(duration=1)
    data = pd.DataFrame({"high": [0, 2, 1], "low": [0, 0, -2], "close": [0, 0, 0]})
    true_labels = pd.DataFrame(
        {
            "positive_label": [2, 1, 0],
            "negative_label": [0, -2, 0],
        }  # positive_label = high - close; negative_label = low - close
    )
    labels = labeler.transform(data)
    for i in range(2):
        for j in range(2):
            assert labels.iloc[i, j] == true_labels.iloc[i, j]
