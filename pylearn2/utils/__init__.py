import pickle
import scipy.misc as misc

from pylearn2.config import yaml_parse
import numpy as np

def read_text(filename):
    with open(filename, 'r') as f:
        txt = f.read()
    f.close()
    return txt

def load_yaml_template(template_file, _dict={}):
    template = read_text(template_file)
    yaml = template % _dict
    obj = yaml_parse.load(yaml)
    return obj

def pickle(obj, filename):
    pickle.dump(obj,open(filename,'w'))

def upickle(filename):
    return pickle.load(open(filename,'r'))

def observation_to_image(observation, start, shape):
    return observation.intArray[start::].reshape(shape[0],shape[1])

def apply_palette(image, palette):
    return palette[image]

def rgb_to_grey(image):
    return np.sqrt(image**2,axis=2)

def resize_image(image, size):
    return misc.imresize(image, size=size)

def crop_image(image, start, size):
    return image[start[0]:start[0]+size[0],start[1]:start[1]+size[1]]
