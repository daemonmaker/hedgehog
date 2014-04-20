"""
Class for managing datasets created online.
"""

import numpy
from pylearn2.datasets import Dataset
import ipdb


class Replay(Dataset):
    def __init__(total_size, img_dims, num_frames, action_dims):
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
            len(img_dims.shape) == 2 and
            img_dims[0] > 0 and
            img_dims[1] > 0
        )
        self.img_dims = img_dims

        assert(num_frames > 0)
        self.num_frames = num_frames

        assert(action_dims > 0)
        self.action_dims = action_dims

        # Allocate memory
        self.phis = numpy.zeros(
            (total_size, img_dims[0], img_dims[1], num_frames)
        )
        self.phi_primes = numpy.zeros(
            (total_size, img_dims[0], img_dims[1], num_frames)
        )
        self.actions = numpy.zeros((total_size, action_dims))
        self.rewards = numpy.zeros(total_size, 1)

        # Setup ring
        self.current_exp = 0
        self.full = False  # Whether the memory is full

    def add(exp):
        """
        exp : tuple
            4-tuple containing phi_t, a_t, r_t, phi_{t+1}
        """
        self.phis[self.current_exp, :] = exp[0]
        self.actions[self.current_exp, :] = exp[1]
        self.rewards[self.current_exp, :] = exp[2]
        self.phi_primes[self.current_exp, :] = exp[3]

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

        # Setup iterator
        if mode == 'sequential' or mode == 'random_uniform':
            total_size = self.total_size
            if not self.full:
                total_size = self.current_record

            # mode is sequential
            if mode == 'sequential':
                self.subset_iterator = SequentialSubsetIterator(
                    self.total_size,
                    batch_size,
                    num_batches,
                    rng=None
                )

            # mode is random uniform
            else:
                self.iter = RandomUniformSubsetIterator(
                    self.total_size,
                    batch_size,
                    num_batches,
                    rng
                )

            return self

        else:
            raise NotImplementedError("Iteration mode '" +
                                      mode + "' not supported.")

    @wraps(Dataset.__iter__)
    def __iter__(self):
        return self

    def next(self):
        slice = self.subset_iterator.next()
        return (
            self.phis[slice],
            self.actions[slice],
            self.rewards[slice],
            self.phi_primes[slice]
        )

if name == '__main__':
    