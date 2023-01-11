import json
import pickle
import copy
import agv_individual
import base64

from agv_xai_utils import *

class ModelLoader(object):

    def __init__(self):
        self.model = None
        self.gradcam_mask_dict = None #new attributo to load selected pixels

    def load(self,path):
        with open(path, 'r') as jfile:
            jmodel = json.load(jfile)
            self.model = copy.copy(jmodel)
            filters_data_byte = base64.b64decode(jmodel["filters_data"])
            self.model["filters_data"] = pickle.loads(filters_data_byte)
        return self

    def save(self,path):
        with open(path, 'w') as jfile:
            jmodel = copy.copy(self.model)
            filters_data_b64 = base64.b64encode(pickle.dumps(self.model["filters_data"]))
            jmodel["filters_data"] = filters_data_b64.decode("ascii")
            json.dump(jmodel, jfile, indent=4)
        return self

    def apply(self,X):
        ilast = 0 
        image = X
        for fid in self.model["filters"]:
            ifilter = self.model["filters_data"][fid]
            image = ifilter(image,*self.model["params"][ilast:ilast+ifilter.nparams()])
            ilast += ifilter.nparams()
        return image

    def to_individual(self):
        indv = agv_individual.Individual(0,0,float("inf"))
        indv.genotype = self.model["filters"]
        indv.params = self.model["params"]
        indv.fitness = self.model["fitness"]
        indv.filters = self.model["filters_data"]
        return indv

    def get_filters(self):
        list_filter = []
        ilast = 0 
        for fid in self.model["filters"]:
            ifilter = self.model["filters_data"][fid]
            list_filter.append((ifilter, *self.model["params"][ilast:ilast+ifilter.nparams()]))
            ilast += ifilter.nparams()
        return list_filter

    def from_individual(self, individual, metainfo = None):
        self.model = {
            "filters" : individual.genotype,
            "params" : individual.params,
            "fitness" : individual.fitness
        }
        if metainfo is not None:
            self.model = { **self.model,  **metainfo }
        #serialize functions
        self.model["filters_data"] = individual.filters
        #retult self
        return self