"""
Basic agent for playing Atari games using Deep ConvNet to estimate Q-function.
Modeled after skeleton example agent in rl-glue python-codec.
"""
__authors__ = ["Dustin Webb", "Tom Le Paine"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb", "Tom Le Paine"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

import collections as col
import copy

import numpy as np

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from random import Random
from hedgehog.pylearn2.datasets.replay import Replay
import hedgehog.pylearn2.utils as utils
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD
from theano import function
import theano.tensor as T
import ipdb


def setup():
    N = 1024  # The paper keeps 1000000 memories
    num_frames = 4  # Prescribed by paper
    img_dims = (84, 84)  # Prescribed by paper
    action_dims = 18  # Prescribed by ALE
    batch_size = 32
    learning_rate = 1e-2
    batches_per_iter = 1  # How many batches to pull from memory

    model_yaml = '../models/model_conv.yaml'
    model = utils.load_yaml_template(model_yaml)

    dataset = Replay(N, img_dims, num_frames, action_dims)

    monitoring_dataset = {}
    monitoring_dataset['train'] = dataset

    algo = SGD(
        batch_size=batch_size,
        learning_rate=learning_rate,
        batches_per_iter=batches_per_iter,
        monitoring_dataset=monitoring_dataset
    )

    train = Train(dataset=dataset, model=model, algorithm=algo)

    view = DeepMindPreprocessor(img_dims)

    return BasicAgent(model, dataset, train, view, k=num_frames)


class DeepMindPreprocessor():
    def __init__(self, img_dims):
        assert(type(img_dims) == tuple and len(img_dims) == 2)
        self.img_dims = img_dims
        self.offset = 128  # Start of image in observation
        self.atari_frame_size = (210, 160)
        self.reduced_frame_size = (110, 84)  # Prescribed by paper
        self.crop_start = (20, 0)  # Crop start prescribed by authors

    def get_frame(self, observation):
        #  TODO confirm this does a deep copy
        image = utils.observation_to_image(
            observation,
            self.offset,
            self.atari_frame_size
        )
        image /= 2  # Calculate idxs into Atari 2600 pallete
        image = utils.resize_image(image, self.reduced_frame_size)
        image = utils.crop_image(image, self.crop_start, self.img_dims)
        return image


class BasicAgent():
    lastAction = Action()
    lastObservation = Observation()

    def __init__(self, model, dataset, train, view, epsilon=0.1, k=4):
        # Validate and store parameters
        assert(model)
        self.model = model

        assert(dataset)
        self.dataset = dataset

        assert(train)
        self.train = train

        assert(view)
        self.view = view

        assert(epsilon > 0 and epsilon <= 1)
        self.epsilon = epsilon

        assert(k > 0)
        self.k = k

    # Init helper member variables
        self.action_count = 0
        self.reward = 0  # Accumulator for reward values

        # Init frame memory
        self.frame_memory = col.deque(maxlen=self.k)

        # Compile action function
        print 'BASIC AGENT: Compiling action function...'
        phi_eq = T.tensor4()
        r = T.fvector('r')
        gamma = T.fscalar('gamma')
        q_eq = self.model.fprop(phi_eq)
        action_eq = T.argmax(q_eq, axis=1)
        self.action_func = function([phi_eq], action_eq)
        print 'BASIC AGENT: Done.'

    def agent_init(self, taskSpec):
        self.lastAction = Action()
        self.lastObservation = Observation()

    def agent_start(self, observation):
        print 'BASIC AGENT: start'
        # Generate random action, one-hot binary vector
        self.select_action()

        self.action_count += 1
        frame = self.view.get_frame(observation)
        self.frame_memory.append(frame)
        return self.action

    def select_action(self, phi=None):
        if self.action_count % self.k == 0:
            action = Action()
            action.intArray = [0]*18
            if (np.random.rand() > self.epsilon) and phi:
                # Get action from Q-function
                #print 'q action...'
                phi = np.array(phi)[:, :, :, None]
                action_int = self.action_func(phi)[0]
                action.intArray[action_int] = 1
            else:
                # Get random action
                action.intArray[np.random.randint(0, 18)] = 1
            self.action = action

    def agent_step(self, reward, observation):
        self.reward += reward

        #print 'BASIC AGENT: step'
        frame = self.view.get_frame(observation)
        self.frame_memory.append(frame)

        if self.action_count == self.k:
            self.reward = 0
            self.phi = np.array(self.frame_memory)

        elif self.action_count % self.k == 0:
            #  Create a new phi
            #  Reset reward to 0
            print 'self.action: ' + str(self.action.intArray)
            self.dataset.add(self.phi, self.action.intArray, self.reward)

            self.reward = 0
            self.phi = np.array(self.frame_memory)

            if self.action_count >= (self.train.algorithm.batch_size*self.k+1):
                ipdb.set_trace()
                self.train.main_loop()

        self.select_action(self.frame_memory)

        self.action_count += 1

        return self.action

    def agent_end(self, reward):
        # TODO What do we do with the reward?
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, msg):
        if msg == 'what is your name?':
            return "Basic Atari agent."
        else:
            return 'I don\'t understand'


def test_agent_step():
    color_range = 128
    size_of_observation = 128+210*160

    agent = setup()

    color = 1
    observation = Observation()
    observation.intArray = np.ones(size_of_observation, dtype=np.uint8)
    observation.intArray *= color
    agent.agent_start(observation)

    for i in range(2, 257):
        reward = float(i)
        color = i
        observation = Observation()
        observation.intArray = np.ones(size_of_observation, dtype=np.uint8)
        observation.intArray *= color

        agent.agent_step(reward, observation)

    ipdb.set_trace()

if __name__ == '__main__':
    agent = setup()
    AgentLoader.loadAgent(agent)
