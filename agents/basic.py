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
import Image
from time import time
import cPickle
from subprocess import call
import glob
import ipdb
# Third-party
import numpy as np
from theano import function
import theano.tensor as T
from rlglue.agent.Agent import Agent # This isn't used?
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



def setup():
    N = 1024  # The paper keeps 1000000 memories
    num_frames = 4  # Prescribed by paper
    img_dims = (84, 84)  # Prescribed by paper
    action_dims = 4  # Prescribed by ALE
    batch_size = 32
    learning_rate = 0.5
    batches_per_iter = 1  # How many batches to pull from memory
    discount_factor = 0.001

    print "Creating action cost."
    action_cost = ActionCost.Action()

    print "Loading model yaml."
    model_yaml = '../models/model_conv.yaml'
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

    print "Creating view."
    view = DeepMindPreprocessor(img_dims)

    print "Creating agent."
    action_map = {
        0: 0,
        1: 1,
        2: 3,
        3: 4,
    }
    return BasicAgent(
        model,
        dataset,
        train,
        view,
        action_map,
        discount_factor=discount_factor,
        k=num_frames
    )


class DeepMindPreprocessor():
    def __init__(self, img_dims):
        assert(type(img_dims) == tuple and len(img_dims) == 2)
        self.img_dims = img_dims
        self.offset = 128  # Start of image in observation
        self.atari_frame_size = (210, 160)
        self.reduced_frame_size = (110, 84)  # Prescribed by paper
        self.crop_start = (20, 0)  # Crop start prescribed by authors

        print "Loading palette."
        self.palette = cPickle.load(open('../stella_palette.pkl', 'rb'))

        self.frame_count = 0

    def get_frame(self, observation):
        #  TODO confirm this does a deep copy
        image = utils.observation_to_image(
            observation,
            self.offset,
            self.atari_frame_size
        )
        image /= 2  # Calculate idxs into Atari 2600 pallete

        # Write out frame
        print('Saving frame...'),
        tic = time()
        image = self.palette[image]
        img_obj = Image.fromarray(image)
        frame = "/Tmp/webbd/drl/frame_%07d.png" % self.frame_count
        self.frame_count += 1
        img_obj.save(open(frame, 'w'))
        toc = time()
        print 'Done. Took %0.2f sec.' % (toc-tic)

        # Resize and crop
        image = np.sqrt(np.sum((image**2), axis=2))
        image = image.astype(np.uint8)
        image = utils.resize_image(image, self.reduced_frame_size)
        image = utils.crop_image(image, self.crop_start, self.img_dims)

        return image


class BasicAgent():
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
    top_score = 0
    model_pickle_path = '/Tmp/webbd/best_model.pkl'

    def __init__(self, model, dataset, train, view, action_map,
                 discount_factor=0.8, epsilon=0.6, k=4):
        # Validate and store parameters
        assert(model)
        self.model = model

        assert(dataset)
        self.dataset = dataset

        assert(train)
        self.train = train

        assert(view)
        self.view = view

        assert(action_map and type(action_map) == dict)
        self.action_map = action_map

        assert(discount_factor > 0)
        if (discount_factor >= 1):
            print "WARNING: Discount factor >= 1, learning may diverge."
        self.discount_factor = discount_factor

        assert(epsilon > 0 and epsilon <= 1)
        self.epsilon = epsilon

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
        self.lastAction = Action()
        self.lastObservation = Observation()

    def agent_start(self, observation):
        print 'BASIC AGENT: start'
        # Generate random action, one-hot binary vector
        self._select_action()

        self.action_count += 1
        frame = self.view.get_frame(observation)
        self.frame_memory.append(frame)
        return self.action

    def _select_action(self, phi=None):
        if self.action_count % self.k == 0:
            if (np.random.rand() > self.epsilon) and phi:
                # Get action from Q-function
                #print 'q action...'
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
            #action.intArray = [0]*18
            #action.intArray[self.action_map[action_int]] = 1
            action.intArray = [self.action_map[action_int]]
            self.action = action

    def agent_step(self, reward, observation):
        self.reward += reward
        self.total_reward += reward

        if self.action_count % 100:
            print "self.total_reward: %d" % self.total_reward

        #print 'BASIC AGENT: step'
        frame = self.view.get_frame(observation)
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

        return self.action

    def agent_train(self, terminal):
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
        phi, action, reward, phi_prime = self.replay.next()
        y = self.y_func(reward, self.discount_factor, phi_prime)[:, None]
        return (phi, action, y)

    def agent_end(self, reward):
        # TODO What do we do with the reward?
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, msg):
        if msg == 'terminal':
            self.terminal = True
            return 'Set terminal.'

        elif msg == 'episode_end':
            self.episode_rewards = self.total_reward

            print 'Episode %d reward: %d' % (self.episode,
                self.episode_rewards)

            # If you get a top score
            if self.episode_rewards > self.top_score:
                self.top_score == self.episode_rewards
                # Log it
                print('Top score achieved! Saving model...'),
                # Save model
                tic = time()
                cPickle.dump(self.model, self.model_pickle_path)
                toc = time()
                print 'Done. Took %0.2f sec.' % (toc-tic)

            self.total_reward = 0
            self.terminal = False

            # Save video
            tic = time()
            video_file = '/Tmp/webbd/episode_%06d.avi' % self.episode
            print("Creating video (%s)..." % video_file),
            ret = call([
                'ffmpeg',
                '-i',
                '/Tmp/webbd/drl/frame_%07d.png',
                video_file
            ])
            toc = time()
            print "Done with status: %d. Took %0.2f sec." % (ret, toc-tic)

            # Remove frames
            tic = time()
            print("Removing frames..."),
            ret = call(['rm'] + glob.glob('/Tmp/webbd/drl/*.png'))
            toc = time()
            print 'Done with status: %d. Took %0.2f sec.' % (ret, toc-tic)

            self.episode += 1
            self.view.frame_count = 0

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

    ipdb.set_trace()

if __name__ == '__main__':
    #test_agent_step()
    agent = setup()
    AgentLoader.loadAgent(agent)
