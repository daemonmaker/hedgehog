"""

"""
__authors__ = ["Dustin Webb", "Tom Le Paine"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb", "Tom Le Paine"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.utils import wraps
import theano.tensor as T
import numpy as np


class Action(DefaultDataSpecsMixin, Cost):
    @wraps(Default.expr)
    def expr(self, model, data, **kwargs):
        X, Y = data
        Y_hat = model.fprop(X)
        one_hot = T.matrix('one_hot')
        return T.sqr(Y - T.max(one_hot*Y_hat, axis=0)).mean()
