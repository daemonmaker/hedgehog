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

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from random import Random
from pylearn2.datasets.replay import Replay
import pylearn2.utils as utils
from theano import function
import theano.tensor as T
import ipdb


class basic_agent():
    lastAction = Action()
    lastObservation = Observation()

    def __init__(self, model_yaml, epsilon=0.1, k=4):
        # Validate and store parameters
        assert(replay_data.__class__.__name__ == 'Replay')
        self.model_yaml = model_yaml
        
        assert(epsilon>0 and epsilon<=1)
        self.epsilon = epsilon

        assert(k>0)
        self.k = k
        
        # Load model yaml
        self.model = utils.load_yaml_template(model_yaml)

        # Set counter
        self.action_count = 0
        
        # Set frame memory
        self.frame_memory = col.deque(maxlen=self.k)

        # Compile action function
        phi_eq = T.tensor4()
        r = T.fvector('r')
        gamma = T.fscalar('gamma')
        q_eq = self.model.fprop(phi_eq)
        action_eq = T.argmax(q_eq,axis=1)
        self.action = function([phi_eq], action_eq)

    def get_frame(observation):
        image = utils.observation_to_image(observation, 128, (210, 160))/2
        image = utils.resize_image(image, (110, 84))
        image = utils.crop_image(image, (20, 0), (84, 84))
        image = image.astype(np.float32)
        return image

    def agent_init(self, taskSpec):
        self.lastAction = Action()
        self.lastObservation = Observation()

    def agent_start(self, observation):
        # Generate random action, one-hot binary vector
        action = self.random_action()

        self.lastAction = copy.deepcopy(action)
        self.lastObservation = copy.deepcopy(observation)

        self.action_count += 1
        frame = self.get_frame(observation)
        self.frame_memory.append(frame)
        return action

    def random_action(self, phi=None):
        action = Action()
        action.intArray = [0]*18
        if (numpy.random.rand() > self.epsilon) or not phi==None:
            phi = np.array(phi)[:,:,:,None]
            action_int = self.action(phi)[0]
            action.intArray[action_int] = 1
        else:
            action.intArray[numpy.random.randint(0, 18)] = 1

        return action

    def agent_step(self, reward, observation):
        frame = self.get_frame(observation)
        self.frame_memory.append(frame)

        if self.action_count % self.k == 0:
            action = self.random_action(self.frame_memory)
        else:
            action = self.lastAction

        #self.lastAction = copy.deepcopy(action)
        self.lastObservation = copy.deepcopy(observation)

        self.action_count +=1
        return action

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

if __name__ == '__main__':
    AgentLoader.loadAgent(Basic())
