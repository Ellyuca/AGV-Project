import sys
import os
import random
from filters import *
from filters_d import *

from transformations_d_v2 import *

#set modules path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(os.path.join(os.path.dirname(__file__),'..', ".."))

class ParameterType(object):
    def __init__(self, strtype):
        self.ptype = eval(strtype) if strtype != 'class' else int
        self.is_class_type = strtype == 'class'

    def test(self, x, domain):
        if type(x) == self.ptype:
            if self.is_class:
                return x in domain
            else:
                return  domain[0] <= x and x <= domain[1]
        return False

    def is_float(self):
        return self.ptype == float

    def is_int(self):
        return self.ptype == int and not self.is_class_type

    def is_class(self):
        return self.is_class_type

class ParameterDomain(object):
    def __init__(self, name, strtype, domain, vinit):
        self.name = name 
        self.ptype = ParameterType(strtype)
        self.domain = domain 
        self.value = vinit
    
    def test(self,x):
        return self.ptype.test(x, self.domain) 

    def random(self):
        if self.ptype.is_float():
            return random.uniform(*self.domain)
        elif self.ptype.is_int():
            return random.randint(*self.domain)
        elif self.ptype.is_class():
            return random.choice(self.domain)

class ImageFilter(object):
    def __init__(self, name, parameters_domains, fun):
        self.name = name
        self.domains = parameters_domains
        self.fun = fun  

    def nparams(self):
        return len(self.domains)

    def __call__(self, image, *params):
        return self.fun(image, *params)



# INSTAGRAM 
Filters = [
    ImageFilter('clarendon', [ 
        ParameterDomain('strenght_s', 'float', [0.7,1.0], 1.0),
        ParameterDomain('intensity_alpha', 'float', [0.9,1.1], 1.0) 
    ], clarendon),
    ImageFilter('gingham', [ 
        ParameterDomain('strenght_s', 'float', [0.7,1.0], 1.0),
        ParameterDomain('intensity_alpha', 'float', [0.9,1.1], 1.0) 
    ], gingham),
    ImageFilter('juno', [ 
        ParameterDomain('strenght_s', 'float', [0.7,1.0], 1.0),
        ParameterDomain('intensity_alpha', 'float', [0.9,1.1], 1.0) 
    ], juno),
    ImageFilter('reyes', [ 
        ParameterDomain('strenght_s', 'float', [0.7,1.0], 1.0),
        ParameterDomain('intensity_alpha', 'float', [0.9,1.1], 1.0) 
    ], reyes),
    ImageFilter('lark', [ 
        ParameterDomain('strenght_s', 'float', [0.7,1.0], 1.0),
        ParameterDomain('intensity_alpha', 'float', [0.9,1.1], 1.0) 
    ], lark_hsv),
    ImageFilter('Hudson', [ 
        ParameterDomain('strenght_s', 'float', [0.7,1.0], 1.0),
        ParameterDomain('intensity_alpha', 'float', [1.0,1.15], 1.0) 
    ], hudson),
    ImageFilter('Slumber', [ 
        ParameterDomain('strenght_s', 'float', [0.7,1.0], 1.0),
        ParameterDomain('intensity_alpha', 'float', [1.0,1.15], 1.0) 
    ], slumber),
    ImageFilter('Stinson', [ 
        ParameterDomain('strenght_s', 'float', [0.7,1.0], 1.0),
        ParameterDomain('intensity_alpha', 'float', [1.0,1.15], 1.0) 
    ], stinson),
    ImageFilter('Rise', [ 
        ParameterDomain('strenght_s', 'float', [0.7,1.0], 1.0),
        ParameterDomain('intensity_alpha', 'float', [1.0,1.15], 1.0) 
    ], rise),
    ImageFilter('Perpetua', [ 
        ParameterDomain('strenght_s', 'float', [0.7,1.0], 1.0),
        ParameterDomain('intensity_alpha', 'float', [1.1,1.45], 1.1) 
    ], perpetua)
]

if __name__ == "__main__":
    #import dataset
    from datasets import CIFAR10Dataset
    #set modules path
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
    sys.path.append(os.path.join(os.path.dirname(__file__),'..', ".."))
    #test
    S = 55
    X, Y = CIFAR10Dataset().get_test_dataset()
    print(X.shape, Y.shape)
    print(X[S].shape)
    show_image(X[S])
    for f in Filters:
        show_image(f(X[S]))