import pickle
from pylearn2.config import yaml_parse


def read_text(filename):
    with open(filename, 'r') as f:
        txt = f.read()
    f.close()
    return txt

def load_yaml_template(template_file, _dict):
    template = read_text(template_file)
    yaml = template % _dict
    obj = yaml_parse.load(yaml)
    return obj

def pickle(obj, filename):
    pickle.dump(obj,open(filename,'w'))

def upickle(filename):
    return pickle.load(open(filename,'r'))
