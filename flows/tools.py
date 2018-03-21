import numpy as np
from sklearn.base import TransformerMixin

class OneHotEncoder(TransformerMixin):
    def fit(self, y):
        classes = np.arange(y.max()+1)
        self._map = (classes) == np.repeat(classes.reshape(-1, 1), len(classes), axis=1)
        return self

    def transform(self, y):
        return self._map[y].astype(np.float)

    def inverse_transform(self, y):
        return np.argmax(y, axis=1)


class CategoricalEncoder(TransformerMixin):
    def fit(self, y):
        self._imap = np.unique(y)
        return self

    def transform(self, y):
        return np.argmax(np.repeat(np.asarray(y).reshape(-1, 1), len(self._imap), axis=1) == self._imap, axis=1)

    def inverse_transform(self, y):
        return self._imap[y]


class IdentityEncoder(TransformerMixin):
    def fit(self, y):
        return self

    def transform(self, y):
        return y

    def inverse_transform(self, y):
        return y


class StackedTransformation(TransformerMixin):
    def __init__(self, trans):
        self._trans = trans
    
    def fit(self, y):
        _y = y
        for T in self._trans:
            _y = T.fit_transform(_y)

        return self
    
    def transform(self, X):
        _y = y
        for T in self._trans:
            _y = T.transform(_y)
        
        return _y
        
    def inverse_transform(self, y):
        _y = y
        for T in reversed(self._trans):
            _y = T.inverse_transform(_y)
        
        return _y