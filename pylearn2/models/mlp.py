"""
Modification of Pylearn2
"""
__authors__ = ["Dustin Webb", "Tom Le Paine"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb", "Tom Le Paine"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

from pylearn2.models.mlp import *
from pylearn2.utils import wraps
from pylearn2.space import VectorSpace
from pylearn2.space import CompositeSpace
import ipdb


class RLMLP(MLP):
    def __init__(self, action_dims=None, **kwargs):
        self.action_dims = action_dims
        super(RLMLP, self).__init__(**kwargs)

    @wraps(MLP.cost_from_X_data_specs)
    def cost_from_X_data_specs(self):
        if self.action_dims:
            space = [self.get_input_space()]
            space.append(VectorSpace(dim=self.action_dims))
            #space.append(self.get_output_space())
            space.append(VectorSpace(dim=1))
            space = CompositeSpace(space)

            source = (
                self.get_input_source(),
                'one_hot_mat',
                self.get_target_source()
            )

            return (space, source)
