"""
Utils

This file provides an implementation of helping classes and functions.
"""

import abc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


class Picklable:
    """
    Simple class for saving (and loading) functionality using pickle.
    """

    def save(self, name: str=None, concat: bool=False):
        """
        Save object using pickle.

        Parameters
        ----------
        name: str
            The filename for the saved object.

        concat: bool
            Whether to add the class name to the file name.         
        """

        if name is None:
            name = self.__class__.__name__
        if concat:
            name = self.__class__.__name__ + "_" + name
        with open(name, "wb") as output:
            pickle.dump(self, output, -1)

    def load(self, name: str=None, concat: bool=False):
        """
        Load object using pickle.

        Parameters
        ----------
        name: str
            The filename for the loaded object.

        concat: bool
            Whether to add the class name to the file name.         
        """

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

    Parameters
    ----------
    title: str
        Title of the plot.

    mode: str
        Determines the drawing mode for this scatter trace. If
        the provided `mode` includes "text" then the `text`
        elements appear at the coordinates. Otherwise, the
        `text` elements appear on hover. If there are less than
        20 points and the trace is not stacked then the default
        is "lines+markers". Otherwise, "lines".
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

    Parameters
    ----------
    X : iterable
        Training data.

    y : iterable
        Training targets.

    downprojector : callable, default=None
        Data downprojection method for visualization.

    title: str
        Title of the plot.    
    """
    if downprojector is not None:
        embedding = downprojector.fit_transform(X)
    else:
        embedding = X.iloc[:, :2].values
    data = pd.DataFrame(embedding, columns=["X Value", "Y Value"], index=X.index)
    data["Category"] = y
    fig = px.scatter(
        data,
        x=data.columns[0],
        y=data.columns[1],
        color=data["Category"],
        hover_name=data.index,
    )
    fig.update_layout(title=title)
    fig.show()