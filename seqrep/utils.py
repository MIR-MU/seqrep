import abc

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


class Picklable:
    """
    Simple class for saving (and loading) functionality
    using pickle.
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
