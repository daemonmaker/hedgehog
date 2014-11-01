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
import logging
#import ipdb

# Third-party
import numpy as np
from theano import function
import theano.tensor as T
from rlglue.agent import AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from pylearn2.train import Train
import pylearn2.monitor as monitor
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.datasets import Dataset
from pylearn2.utils import wraps
from pylearn2.termination_criteria import EpochCounter
from pylearn2.space import CompositeSpace
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.training_algorithms.learning_rule import RPROP

# Internal
from hedgehog.pylearn2.datasets.replay import Replay
import hedgehog.pylearn2.utils as utils
import hedgehog.pylearn2.costs.action as ActionCost
import hedgehog.percept_processors.percept_preprocessors as ppp

log = logging.getLogger(__name__)
logging.basicConfig(filename='basic.log', level=logging.DEBUG)


def setup():
    N = 300000  # The paper keeps 1,000,000 memories
    num_frames = 4  # Prescribed by paper
    img_dims = (84, 84)  # Prescribed by paper
    action_dims = 4  # Prescribed by ALE
    batch_size = 32
    learning_rate = 0.05
    batches_per_iter = 1  # How many batches to pull from memory
    discount_factor = 0.95
    base_dir = '/data/lisa/exp/webbd/drl/experiments/2014-11-01'
    model_pickle_path = os.path.join(base_dir, 'best_model.pkl')

    log.info("Creating action cost.")
    action_cost = ActionCost.Action()

    # Load the model if it exists
    if os.path.exists(model_pickle_path):
        model = cPickle.load(open(model_pickle_path, 'rb'))
        model = monitor.push_monitor(model, "at",  transfer_experience=True)

    # Otherwise create a new model
    else:
        # TODO This is a hacky way to find the model yaml
        model_yaml = os.path.dirname(os.path.realpath(__file__))
        model_yaml = os.path.join(model_yaml, '../models/model_conv.yaml')
        log.info("Loading model yaml (%s)" % model_yaml)
        yaml_params = {
            'num_channels': num_frames,
            'action_dims': action_dims,
        }
        model = utils.load_yaml_template(model_yaml, yaml_params)

    log.info("Creating dataset.")
    dataset = Replay(N, img_dims, num_frames, action_dims)

    #monitoring_dataset = {}
    #monitoring_dataset['train'] = dataset

    log.info("Creating terminiation criterion.")
    termination_criterion = EpochCounter(1)

    log.info("Creating training algorithm.")
    algo = SGD(
        batch_size=batch_size,
        learning_rate=learning_rate,
        batches_per_iter=batches_per_iter,
        #monitoring_dataset=monitoring_dataset
        monitoring_dataset=None,
        cost=action_cost,
        termination_criterion=termination_criterion,
        learning_rule=RPROP()
    )

    log.info("Creating training object.")
    train = Train(dataset=None, model=model, algorithm=algo)

    log.info("Creating percept_preprocessor.")
    percept_preprocessor = ppp.DeepMindPreprocessor(img_dims, base_dir)

    log.info("Creating agent.")
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
        model_pickle_path,
        discount_factor=discount_factor,
        k=num_frames,
        epsilon=1,
        epsilon_anneal_frames=5000000
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
    model_pickle_path: string
        Path to model pickle file.
    save_rate: int
        Number of episodes after which model should be pickled and video
        created.
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
    all_time_total_frames = 0
    episode_training_time = 0
    episode_start = time()
    train_setup = 0
    train_reward = 0

    def __init__(
        self, model, dataset, train, percept_preprocessor, action_map,
        base_dir, model_pickle_path, save_rate=100,
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

        assert(os.path.exists(base_dir))
        self.base_dir = base_dir

        assert(os.path.exists(os.path.dirname(model_pickle_path)))
        self.model_pickle_path = model_pickle_path

        assert(save_rate > 0)
        self.save_rate = save_rate

        assert(discount_factor > 0)
        if (discount_factor >= 1):
            log.warning("Discount factor >= 1, learning may diverge.")
        self.discount_factor = discount_factor

        assert(epsilon >= 0 and epsilon <= 1)
        self.epsilon = epsilon

        assert(epsilon_anneal_frames >= 0)
        self.epsilon_anneal_frames = epsilon_anneal_frames

        assert(epsilon_end >= 0)
        self.epsilon_end = epsilon_end

        self.epsilon_annealing_rate = 0
        if self.epsilon_anneal_frames > 0:
            self.epsilon_annealing_rate = float(self.epsilon - self.epsilon_end)
            self.epsilon_annealing_rate /= float(self.epsilon_anneal_frames)
        log.info('Epsilon annealing rate: %0.10f' % self.epsilon_annealing_rate)

        assert(k > 0)
        self.k = k

        self.train.dataset = self

        # Init helper member variables
        self.action_count = 0
        self.reward = 0  # Accumulator for reward values

        # Init frame memory
        self.frame_memory = col.deque(maxlen=self.k)

        # Compile action function
        log.info('BASIC AGENT: Compiling action function...'),
        phi_eq = T.tensor4()
        q_eq = self.model.fprop(phi_eq)
        action_eq = T.argmax(q_eq, axis=1)
        self.action_func = function([phi_eq], action_eq)
        log.info('Done.')

        # Compile max q
        log.info('BASIC AGENT: Compiling y function...'),
        max_action_eq = T.max(q_eq, axis=1)
        self.max_q_func = function([phi_eq], max_action_eq)
        log.info('Done.')

        # Compile maximum action function
        log.info('BASIC AGENT: Compiling y function...'),
        r = T.fvector('r')
        gamma = T.fscalar('gamma')
        y = r + gamma*max_action_eq
        self.y_func = function([r, gamma, phi_eq], y)
        log.info('Done.')

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
        log.info('BASIC AGENT: start')
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
                log.info("self.epoch_q_values: %s" % str(self.epoch_q_values))
                log.info("self.epoch_rewards: %s" % str(self.epoch_rewards))
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
        if self.action_count >= ((self.train.algorithm.batch_size+1)*self.k+1):
            tic = time()
            if self.train_setup == 0:
                self.train.main_loop()

                data_specs = self.train.algorithm.cost.get_data_specs(
                    self.model)

                # The iterator should be built from flat data specs, so it
                # returns flat, non-redundent tuples of data.
                mapping = DataSpecsMapping(data_specs)
                space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
                source_tuple = mapping.flatten(
                    data_specs[1],
                    return_tuple=True
                )
                if len(space_tuple) == 0:
                    # No data will be returned by the iterator, and it is
                    # impossible to know the size of the actual batch. It
                    # is not decided yet what the right thing to do should be.
                    raise NotImplementedError(
                        "Unable to train with SGD, because the cost does not"
                        " actually use data from the data set. "
                        "data_specs: %s" % str(data_specs)
                    )
                flat_data_specs = (CompositeSpace(space_tuple), source_tuple)
                self.flat_data_specs = flat_data_specs
                self.train_setup = 1

            else:
                tic_iter = time()
                temp_iter = self.train.dataset.iterator(
                    mode=self.train.algorithm.train_iteration_mode,
                    batch_size=self.train.algorithm.batch_size,
                    data_specs=self.flat_data_specs,
                    return_tuple=True,
                    rng=self.train.algorithm.rng,
                    num_batches=self.train.algorithm.batches_per_iter
                )
                toc_iter = time()
                log.debug('Iter creation time: %0.2f' % (toc_iter - tic_iter))

                tic_next = time()
                batch = temp_iter.next()
                toc_next = time()
                log.debug('Iter next time: %0.2f' % (toc_next - tic_next))

                tic_sgd = time()
                self.train.algorithm.sgd_update(*batch)
                toc_sgd = time()
                log.debug('SGD time: %0.2f' % (toc_sgd - tic_sgd))

                log.info('Frames seen: %d' % self.all_time_total_frames)
                log.info('Epsilon: %0.10f' % self.epsilon)

            toc = time()
            self.episode_training_time += toc-tic
            log.debug('Real train time: %0.2f' % (toc-tic))

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
        # tic = time()
        phi, action, reward, phi_prime = self.replay.next()
        # toc = time()

        # log.debug('Replay next time: %0.2f' % (toc-tic))
        self.train_reward += np.sum(reward)

        log.debug('Train reward: %0.2f' % self.train_reward)
        y = self.y_func(reward, self.discount_factor, phi_prime)[:, None]
        return (phi, action, y)

    def get_num_examples(self):
        return 0

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

            # Calculate values for logging
            high_score_string = ''
            param = np.mean(self.model.layers[0].get_params()[0].get_value())

            toc = time()
            episode_time = toc - self.episode_start

            # If you get a top score
            time_to_save = self.episode % self.save_rate == 0
            high_score_reached = self.total_reward >= self.top_score

            if high_score_reached:
                high_score_string = 'High score!'
                self.top_score = self.total_reward

            log.info('--Episode %d--' % self.episode)
            log.info(
                'Score: %d - %s' % (self.total_reward, high_score_string)
            )
            log.info('Time: %0.2f sec.' % episode_time)
            log.info(
                'Training time: %0.2f sec.' % self.episode_training_time
            )
            log.info('Training reward: %d' % int(self.train_reward))
            log.info('Parameter mean: %0.5e' % param)

            # Reset logging values
            self.train_reward = 0
            self.episode_start = time()
            self.episode_training_time = 0

            if time_to_save or high_score_reached:
                # Save model
                log.info("Saving model (%s))..." % self.model_pickle_path),
                tic = time()
                cPickle.dump(self.model, open(self.model_pickle_path, 'wb'))
                toc = time()
                log.info('Done. Took %0.2f sec.' % (toc-tic))

                video_name = 'episode_%06d.avi' % self.episode
                self.percept_preprocessor.write_percepts()
                self.percept_preprocessor.save_video(video_name)

            # Reset relevant variables
            self.percept_preprocessor.reset_frames()
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
