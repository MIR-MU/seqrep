import abc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


class Picklable:
    """
    Simple class for saving (and loading) functionality using pickle.
    """

    def save(self, name=None, concat=False):
        if name is None:
            name = self.__class__.__name__
        if concat:
            name = self.__class__.__name__ + "_" + name
        with open(name, "wb") as output:
            pickle.dump(self, output, -1)

    def load(self, name=None, concat=False):
        if name is None:
            name = self.__class__.__name__
        if concat:
            name = self.__class__.__name__ + "_" + name
        with open(name, "rb") as input:
            return pickle.load(input)


class Visualizable(abc.ABC):
    """
    A simple abstract class requiring the implementation of a visualize function.
    """

    @abc.abstractmethod
    def visualize(self):
        """
        This function visualize the outputs or state of the object.
        """
        raise NotImplementedError


def visualize_labels(
    labels, title="Visualization of labels", mode: str = "lines"
) -> None:
    """
    Plot labels.
    """
    if len(labels.shape) == 1:
        labels = pd.DataFrame(labels, columns=["labels"])
    fig = go.Figure()
    fig.update_layout(title=title)
    fig.update_yaxes(title_text="labels")
    for i in range(labels.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=labels.index,
                y=labels.iloc[:, i],
                name=labels.columns[i],
                mode=mode,
            )
        )
    fig.show()


def visualize_data(
    X, y, downprojector=None, title: str = "Visualization of data"
) -> None:
    """
    Plot data in 2D.
    """
    if downprojector is not None:
        embedding = downprojector.fit_transform(X)
        data = pd.DataFrame(embedding, columns=["X Value", "Y Value"], index=X.index)
    else:
        embedding = X.iloc[:, :2].copy()
        data = embedding
        data.columns = ["X Value", "Y Value"]
    data["Category"] = y
    fig = px.scatter(
        data, x="X Value", y="Y Value", color=data["Category"], hover_name=data.index
    )
    fig.show()
