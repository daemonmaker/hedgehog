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

# Standard
import collections as col
from time import time
import cPickle
import os

# Third-party
import numpy as np
from theano import function
import theano.tensor as T
from rlglue.agent.Agent import Agent  # This isn't used?
from rlglue.agent import AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.datasets import Dataset
from pylearn2.utils import wraps
from pylearn2.termination_criteria import EpochCounter

# Internal
from hedgehog.pylearn2.datasets.replay import Replay
import hedgehog.pylearn2.utils as utils
import hedgehog.pylearn2.costs.action as ActionCost
import hedgehog.percept_processors.percept_preprocessors as ppp


def setup():
    N = 10000  # The paper keeps 1000000 memories
    num_frames = 4  # Prescribed by paper
    img_dims = (84, 84)  # Prescribed by paper
    action_dims = 4  # Prescribed by ALE
    batch_size = 32
    learning_rate = 0.5
    batches_per_iter = 1  # How many batches to pull from memory
    discount_factor = 0.001
    base_dir = '/Tmp/webbd/drl/experiments/2'

    print "Creating action cost."
    action_cost = ActionCost.Action()

    # TODO This is a hacky way to find the model yaml
    model_yaml = os.path.dirname(os.path.realpath(__file__))
    model_yaml = os.path.join(model_yaml, '../models/model_conv.yaml')
    print "Loading model yaml (%s)" % model_yaml
    yaml_params = {
        'num_channels': num_frames,
        'action_dims': action_dims,
    }
    model = utils.load_yaml_template(model_yaml, yaml_params)

    print "Creating dataset."
    dataset = Replay(N, img_dims, num_frames, action_dims)

    #monitoring_dataset = {}
    #monitoring_dataset['train'] = dataset

    print "Creating terminiation criterion."
    termination_criterion = EpochCounter(1)

    print "Creating training algorithm."
    algo = SGD(
        batch_size=batch_size,
        learning_rate=learning_rate,
        batches_per_iter=batches_per_iter,
        #monitoring_dataset=monitoring_dataset
        monitoring_dataset=None,
        cost=action_cost,
        termination_criterion=termination_criterion
    )

    print "Creating training object."
    train = Train(dataset=None, model=model, algorithm=algo)

    print "Creating percept_preprocessor."
    percept_preprocessor = ppp.DeepMindPreprocessor(img_dims, base_dir)

    print "Creating agent."
    action_map = {
        0: 0,
        1: 1,
        2: 3,
        3: 4,
    }
    return BasicQAgent(
        model,
        dataset,
        train,
        percept_preprocessor,
        action_map,
        base_dir,
        discount_factor=discount_factor,
        k=num_frames
    )


class BasicQAgent(object):
    """
    Basic RL Q-learning agent.

    model: Model
        Pylearn2 model object.
        Pylearn2 dataset for storing memories.
    dataset: Dataset
    train: Train
        Pylearn2 train object for training agent.
    percept_preprocessor: PerceptPreprocessor
        Object for preprocessing percepts.
    action_map: dictionary
        Dictionary for mapping actions to ALE actions.
    base_dir: string
        Base directory for storing models, videos, and the like.
    discount_factor: float
        Discount factor for Bellman equation.
    epsilon_start: float
        Rate at which to select random actions.
        0 => Always us policy.
        1 => Always select randomly.
    epsilon_anneal_frames: int
        Number of frames over which epsilon should be annealed. Set to 0 to
        prevent annealing.
    epsilon_end: float
        Final epsilon value. Only valid when epsilon_anneal_frames > 0.
    k: int
        Number of percepts to group into one memory.
    """
    lastAction = Action()
    lastObservation = Observation()
    total_reward = 0
    terminal = False
    minibatches_trained = 0
    minibatch_q_values = []
    epoch_q_values = []
    epoch_rewards = []
    episode_rewards = []
    episode = 0
    epoch_size = 50000
    top_score = -1
    model_pickle_name = 'best_model.pkl'
    all_time_total_frames = 0

    def __init__(
        self, model, dataset, train, percept_preprocessor, action_map,
        base_dir,
        epsilon=1, epsilon_anneal_frames=1000000, epsilon_end=0.1,
        discount_factor=0.8, k=4,
    ):
        # Validate and store parameters
        assert(model)
        self.model = model

        assert(dataset)
        self.dataset = dataset

        assert(train)
        self.train = train

        assert(percept_preprocessor)
        self.percept_preprocessor = percept_preprocessor

        assert(action_map and type(action_map) == dict)
        self.action_map = action_map

        assert(base_dir)
        self.base_dir = base_dir

        assert(discount_factor > 0)
        if (discount_factor >= 1):
            print "WARNING: Discount factor >= 1, learning may diverge."
        self.discount_factor = discount_factor

        assert(epsilon >= 0 and epsilon <= 1)
        self.epsilon = epsilon

        assert(epsilon_anneal_frames >= 0)
        self.epsilon_anneal_frames = epsilon_anneal_frames

        assert(epsilon_end >= 0)
        self.epsilon_end = epsilon_end

        if self.epsilon_anneal_frames > 0:
            self.epsilon_annealing_rate = (self.epsilon - self.epsilon_end)
            self.epsilon_annealing_rate /= self.epsilon_anneal_frames

        assert(k > 0)
        self.k = k

        self.train.dataset = self

        # Init helper member variables
        self.action_count = 0
        self.reward = 0  # Accumulator for reward values

        # Init frame memory
        self.frame_memory = col.deque(maxlen=self.k)

        # Compile action function
        print('BASIC AGENT: Compiling action function...'),
        phi_eq = T.tensor4()
        q_eq = self.model.fprop(phi_eq)
        action_eq = T.argmax(q_eq, axis=1)
        self.action_func = function([phi_eq], action_eq)
        print 'Done.'

        # Compile max q
        print('BASIC AGENT: Compiling y function...'),
        max_action_eq = T.max(q_eq, axis=1)
        self.max_q_func = function([phi_eq], max_action_eq)
        print 'Done.'

        # Compile maximum action function
        print('BASIC AGENT: Compiling y function...'),
        r = T.fvector('r')
        gamma = T.fscalar('gamma')
        y = r + gamma*max_action_eq
        self.y_func = function([r, gamma, phi_eq], y)
        print 'Done.'

    def agent_init(self, taskSpec):
        """
        Initializes agent.

        taskSpec: string
            Currently unused. Required by RL-Glue agent interface.
        """
        self.lastAction = Action()
        self.lastObservation = Observation()

    def agent_start(self, observation):
        """
        Starts agent.

        observation: Observation
            Initial RL-Glue observation.
        """
        print 'BASIC AGENT: start'
        self.all_time_total_frames += 1

        # Generate random action, one-hot binary vector
        self._select_action()

        self.action_count += 1
        frame = self.percept_preprocessor.get_frame(observation)
        self.frame_memory.append(frame)
        return self.action

        self._anneal_parameters()

    def agent_step(self, reward, observation):
        """
        Calculates agent's next action.

        reward: float
            Reward received for previous action.
        observation: Observation
            Observation of current state.
        """
        self.all_time_total_frames += 1
        self.reward += reward
        self.total_reward += reward

        if self.action_count % 100:
            print "self.total_reward: %d" % self.total_reward

        #print 'BASIC AGENT: step'
        frame = self.percept_preprocessor.get_frame(observation)
        self.frame_memory.append(frame)

        if self.action_count == self.k:
            self.reward = 0
            self.phi = np.array(self.frame_memory)

        elif self.action_count % self.k == 0:
            #  Normalize reward
            self.reward = np.sign(self.reward)
            #  Add to dataset
            self.dataset.add(self.phi, self.cmd, self.reward)
            #  Create a new phi
            #  Reset reward to 0
            self.reward = 0
            self.phi = np.array(self.frame_memory)

            self.agent_train(self.terminal)
            self.terminal = False

            self.minibatches_trained += 1
            phi = self.phi[:, :, :, None]
            self.minibatch_q_values.append(self.max_q_func(phi)[0])
            if self.minibatches_trained % self.epoch_size == 0:
                self.epoch_q_values.append(np.mean(self.minibatch_q_values))
                self.epoch_rewards.append(np.mean(self.episode_rewards))
                print "self.epoch_q_values:", str(self.epoch_q_values)
                print "self.epoch_rewards:", str(self.epoch_rewards)
                self.episode_rewards = []
                self.minibatch_q_values = []

        self._select_action(self.frame_memory)

        self.action_count += 1

        self._anneal_parameters()

        return self.action

    def _select_action(self, phi=None):
        """
        Utility function for selecting an action.

        phi: ndarray
            Memory from which action should be selected.
        """
        if self.action_count % self.k == 0:
            if (np.random.rand() > self.epsilon) and phi:
                # Get action from Q-function
                phi = np.array(phi)[:, :, :, None]
                action_int = self.action_func(phi)[0]
            else:
                # Get random action
                action_int = np.random.randint(0, len(self.action_map))

            self.cmd = [0]*len(self.action_map)
            self.cmd[action_int] = 1

            # Map cmd to ALE action
            # 18 is the number of commands ALE accepts
            action = Action()
            action.intArray = [self.action_map[action_int]]
            self.action = action

    def _anneal_parameters(self):
        if self.all_time_total_frames < self.epsilon_anneal_frames:
            self.epsilon -= self.epsilon_annealing_rate

    def agent_train(self, terminal):
        """
        Training function.

        terminal: boolean
            Whether current state is a terminal state.
        """
        # Wait until we have enough data to train
        if self.action_count >= (self.train.algorithm.batch_size*self.k+1):
            self.train.main_loop()

    @wraps(Dataset.__iter__)
    def __iter__(self):
        return self

    @wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=False, rng=None,
                 # TODO Remove following options when dataset contract is
                 # corrected. Currently they are not required but is required
                 # by FiniteDatasetIterator.
                 data_specs=None, return_tuple=False):
        self.replay = self.dataset.iterator(
            mode,
            batch_size,
            num_batches,
            topo,
            targets,
            rng
        )
        return self

    def next(self):
        """
        Returns next object from iterator.
        """
        phi, action, reward, phi_prime = self.replay.next()
        y = self.y_func(reward, self.discount_factor, phi_prime)[:, None]
        return (phi, action, y)

    def agent_end(self, reward):
        """
        Call to notify agent that episode is ending.

        reward: float
            Final reward.
        """
        # TODO What do we do with the reward?
        pass

    def agent_cleanup(self):
        """
        Permits agent to cleanup after an episode.
        """
        pass

    def agent_message(self, msg):
        """
        RL-Glue generic message interface.

        msg: string
            Arbitrary message from another process. Current supported values:
            -terminal: Notifies the agent that the next state is a terminal
             state.
            -episode_end: Notifies the agent that the episode has completed.
            -reset: Resets the agent
        """
        if msg == 'terminal':
            self.terminal = True
            return 'Set terminal.'

        elif msg == 'episode_end':
            self.episode_rewards.append(self.total_reward)

            print('Episode %d reward: %d' % (self.episode,
                                             self.total_reward)),

            # If you get a top score
            if self.total_reward > self.top_score:
                self.top_score = self.total_reward

                # Log it
                print(" Top score achieved!")

                # Save model
                print("Saving model..."),
                tic = time()
                file_name = os.path.join(
                    self.base_dir,
                    self.model_pickle_name
                )
                cPickle.dump(self.model, open(file_name, 'w'))
                toc = time()
                print 'Done. Took %0.2f sec.' % (toc-tic)

                video_name = 'episode_%06d.avi' % self.episode
                self.percept_preprocessor.save_video(video_name)

            print ""  # Print newline and carriage return

            # Reset relevant variables
            self.total_reward = 0
            self.terminal = False

            # Count episode
            self.episode += 1

        elif msg == 'reset':
            self.terminal = False
            self.total_reward = 0
            self.action_count = 0
            return 'Reset.'


def test_agent_step():
    print "Testing."
    color_range = 128
    size_of_observation = 128+210*160

    print "Setting up agent."
    agent = setup()

    color = 1
    observation = Observation()
    observation.intArray = np.ones(size_of_observation, dtype=np.uint8)
    observation.intArray *= color
    agent.agent_start(observation)
    agent.agent_train(False)

    for i in range(2, 256):
        print "Round %d" % i
        reward = float(i)
        color = i
        observation = Observation()
        observation.intArray = np.ones(size_of_observation, dtype=np.uint8)
        observation.intArray *= color

        agent.agent_step(reward, observation)
        agent.agent_train(False)

    reward = float(i)
    color = i
    observation = Observation()
    observation.intArray = np.ones(size_of_observation, dtype=np.uint8)
    observation.intArray *= color

    agent.agent_step(reward, observation)

    agent.agent_train(True)

    #ipdb.set_trace()

if __name__ == '__main__':
    #test_agent_step()
    agent = setup()
    AgentLoader.loadAgent(agent)
