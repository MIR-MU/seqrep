try:
    import cPickle as pickle
except ModuleNotFoundError:
     import pickle

# import icecream as ic

class Picklable():
    def save(self, name=None, concat=False):
        if name is None:
            name = self.__class__.__name__
        if concat:
            name = self.__class__.__name__ + '_' + name
        # ic(name)
        with open(name, 'wb') as output:
            pickle.dump(self, output, -1)
        
    def load(self, name=None, concat=False):
        if name is None:
            name = self.__class__.__name__
        if concat:
            name = self.__class__.__name__ + '_' + name
        # ic(name)
        with open(name, 'rb') as input:
            return pickle.load(input)