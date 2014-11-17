import numpy
from matplotlib import pyplot
import theano
import theano.tensor as T

from model import ReinforcementModel

def random_action(num_actions=1):
    action_dims = 4
    p = 1.0/action_dims
    action = numpy.float32(numpy.random.multinomial(1, [0.25, 0.25, 0.25, 0.25], size=(num_actions)))
    return action

model = ReinforcementModel('gah', 'gah', learning_rate=1e-3)

# Sad world emulator
# No matter what you do you get no reward.
r = numpy.zeros(32, dtype=numpy.float32)
gamma = numpy.float32(0.0) # In this bleak world it is better if you have no hope (gamma=0)

# Before learning
data = numpy.float32(numpy.random.rand(4, 84, 84, 32))
gah = model.prediction(data)
pyplot.plot(gah)
gah2 = model.max_q(data)
pyplot.plot(gah2, 'black')

# Learning
for i in range(100):
    # See some stuff
    data = numpy.float32(numpy.random.rand(4, 84, 84, 32))
    # Take a random action
    action = random_action(32)
    # Get no reward
    y = model.y(data, r, gamma)
    # Learn that no matter what you do, there will be no reward
    cost = model.train(data, action, y)
    print cost

# After learning
gah = model.prediction(data)
pyplot.plot(gah)
gah2 = model.max_q(data)
pyplot.plot(gah2, 'black')
