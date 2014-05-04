"""
Class for managing datasets created online.
"""
__authors__ = ["Dustin Webb"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb", "Tom Le Paine"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

import numpy as np
from pylearn2.utils import wraps
from pylearn2.datasets import Dataset
import pylearn2.utils.iteration as iteration
import ipdb
import theano
from time import time
import logging

log = logging.getLogger(__name__)
logging.basicConfig(filename='basic.log',level=logging.DEBUG)

class Replay(Dataset):
    #self.stochastic = False

    def __init__(self, total_size, img_dims, num_frames, action_dims):
        """
        total_size : int
            The total number of records to retain.
        img_dims : tuple
            Three-tuple specifying the images dimensions (x, y).
        num_frames : int
            Number of frames stored together.
        action_dims : int
            Number of possible actions.
        """
        # Validate and store parameters
        assert(total_size > 0)
        self.total_size = total_size

        assert(
            type(img_dims) == tuple and
            len(img_dims) == 2 and
            img_dims[0] > 0 and
            img_dims[1] > 0
        )
        self.img_dims = img_dims

        assert(num_frames > 0)
        self.num_frames = num_frames

        assert(action_dims > 0)
        self.action_dims = action_dims

        # Allocate memory
        self.phis = [0]*total_size
        for i in xrange(total_size):
            self.phis[i] = np.zeros(
            (num_frames, img_dims[0], img_dims[1]),
            dtype=theano.config.floatX
            )

        self.actions = np.zeros(
            (total_size, action_dims),
            dtype=theano.config.floatX
        )

        self.rewards = np.zeros((total_size, 1), dtype=theano.config.floatX)
        self.idxs = np.arange(total_size)

        # Setup ring
        self.current_exp = 0
        self.full = False  # Whether the memory is full

    def can_sample(self):
        return self.full or (self.current_exp > 1)

    def add(self, phi, action, reward):
        """
        phi_t, a_t, r_t, phi_{t+1}
        """
        self.phis[self.current_exp] = phi
        self.actions[self.current_exp, :] = action
        self.rewards[self.current_exp, :] = reward

        self.current_exp += 1
        if self.current_exp >= self.total_size:
            self.current_exp = 0
            self.full = True

        # TODO Does this need to invalidate any existing iterators?

    @wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=False, rng=None):
        # Store parameters for diagnostic purposes
        self.mode = mode
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.topo = topo
        self.targets = targets
        self.rng = rng

        self.phis_temp = np.zeros((self.num_frames,
            self.img_dims[0],
            self.img_dims[1],
            batch_size),
            dtype=theano.config.floatX)

        self.phis_prime_temp = np.zeros((self.num_frames,
            self.img_dims[0],
            self.img_dims[1],
            batch_size),
            dtype=theano.config.floatX)

        total_size = self.total_size - 1
        if not self.full:
            total_size = self.current_exp
        total_size -= 1

        # Setup iterator
        if mode == 'sequential':
            self.iter = iteration.SequentialSubsetIterator(
                total_size,
                batch_size,
                num_batches,
                rng=None
            )

        elif mode == 'shuffled_sequential':
            self.iter = iteration.ShuffledSequentialSubsetIterator(
                total_size,
                batch_size,
                num_batches,
                rng=rng
            )
        elif mode == 'random_slice':
            self.iter = iteration.RandomSliceSubsetIterator(
                total_size,
                batch_size,
                num_batches,
                rng=rng
            )
        elif mode == 'random_uniform':
            self.iter = iteration.RandomUniformSubsetIterator(
                total_size,
                batch_size,
                num_batches,
                rng
            )

        else:
            raise Exception("Unknown iteration mode.")

        return self

    @wraps(Dataset.__iter__)
    def __iter__(self):
        return self

    def next(self):
        slice = self.iter.next()
        ids = self.idxs[slice].copy()

        if self.full:
            ids += self.current_exp
            ids %= self.total_size

        phi_prime_ids = (ids+1) % self.total_size

        for i,_id in enumerate(ids):
            self.phis_temp[:,:,:,i] = self.phis[_id]

        for i,_id in enumerate(phi_prime_ids):
            self.phis_prime_temp[:,:,:,i] = self.phis[_id]

        actions = self.actions[ids]
        rewards = self.rewards[ids].flatten()

        return (
            self.phis_temp,
            actions,
            rewards,
            self.phis_prime_temp)

    # TODO Remove this when dataset contract is corrected. Currently it is not
    # required but is required by FiniteDatasetIterator.
    def get_design_matrix(self, topo=None):
        return self.phis


def test_iter():
    n = 10
    img_dims = (2, 2)
    frames = 2
    action_dims = 1
    r = Replay(n, img_dims, frames, action_dims)

    print 'Partially filling memory.'
    mem_count = n-5
    max_idx = mem_count - 1
    for i in range(mem_count):
        x, y = img_dims
        phi = np.ones((x, y, frames))*i
        action = i
        reward = i
        r.add(phi, action, reward)

    print 'Testing sequential iteration when not full.'
    iter = r.iterator(mode='sequential', batch_size=1)
    for i, (phi, action, reward, phi_prime) in enumerate(iter):
        if action >= max_idx:
            import ipdb
            ipdb.set_trace()

    print 'Testing random iteration when not full.'
    iter = r.iterator(mode='random_uniform', batch_size=1, num_batches=100)
    for i, (phi, action, reward, phi_prime) in enumerate(iter):
        if action >= max_idx:
            import ipdb
            ipdb.set_trace()

    print 'Filling memory past capacity.'
    r = Replay(n, img_dims, frames, action_dims)
    for i in range(n + 3):
        x, y = img_dims
        phi = np.ones((x, y, frames))*i
        action = i
        reward = i
        r.add(phi, action, reward)

    print 'Testing overfull memory has been overwritten. With sequential iter.'
    iter = r.iterator(mode='sequential', batch_size=1)
    for i, (phi, action, reward, phi_prime) in enumerate(iter):
        current_phi_prime = phi_prime[0][0][0][0]
        if current_phi_prime == r.current_exp:
            import ipdb
            ipdb.set_trace()

    print 'Testing overfull memory has been overwritten. With random iter.'
    iter = r.iterator(mode='random_uniform', batch_size=1, num_batches=100)
    for i, (phi, action, reward, phi_prime) in enumerate(iter):
        current_phi_prime = phi_prime[0][0][0][0]
        if current_phi_prime == r.current_exp:
            import ipdb
            ipdb.set_trace()

    print 'Tests executed successfully.'

if __name__ == '__main__':
    test_iter()
