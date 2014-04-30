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
from pylearn2.utils import wraps
import ipdb


class Action(DefaultDataSpecsMixin, Cost):
    @wraps(Cost.expr)
    def expr(self, model, data, **kwargs):
        X, one_hot, Y = data
        Y_hat = model.fprop(X)
        expr = T.sqr(Y - T.max(one_hot*Y_hat, axis=1).dimshuffle(0, 'x'))
        return expr.mean()

    @wraps(Cost.get_data_specs)
    def get_data_specs(self, model):
        return model.cost_from_X_data_specs()
