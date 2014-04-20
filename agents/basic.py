"""
Basic agent for playing Atari games using Deep ConvNet to estimate Q-function.
Modeled after skeleton example agent in rl-glue python-codec.
"""
__authors__ = ["Dustin Webb", "Thomas Paine"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb", "Thomas Paine"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from random import Random
from pylearn2.datasets.replay import Replay
import ipdb


class basic_agent():
    lastAction = Action()
    lastObservation = Observation()

    def __init__(self, model_yaml, epsilon=0.1):
        assert(replay_data.__class__.__name__ == 'Replay')
        self.model_yaml = model_yaml
        self.epsilon = epsilon
        # TODO load yaml
        # TODO compile action function

    def agent_init(self, taskSpec):
        self.lastAction = Action()
        self.lastObservation = Observation()

    def aget_start(self, observation):
        # Generate random action, one-hot binary vector
        action = self.random_action(observation)

        lastAction = copy.deepcopy(action)
        lastObservation = copy.deepcopy(observation)

        return action

    def random_action(self, observation):
        action = Action()
        if numpy.random.rand() > self.epsilon:
            # TODO pick optimal action
        else:
            action.intArray = [0]*18
            action.intArray[numpy.random.randint(0, 18)] = 1

        return action

    def agent_step(self, reward, observation):
        action = self.random_action(observation)

        self.lastAction = copy.deepcopy(action)
        self.lastObservation = copy.deepcopy(observation)

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
