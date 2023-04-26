import os
import random
import copy 
import itertools
import numpy as np
import dill as serialize_and_deserialize

from tqdm import tqdm
from log import Log
from sklearn.utils import shuffle as sk_shuffle
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from nsga2 import nsga_2_pass,dominates
from agv_individual import Individual
from agv_model_loader import ModelLoader

import matplotlib.pyplot as plt

import json

from agv_tests import mkdir_p
from agv_tests import get_cam

from fitness import inv_attack_rate
from agv_datasets import build_model

import math 
STATE_EXT = "state"
JSON_EXT = "json"

def ga_concat(x):
    return list(itertools.chain.from_iterable(x))

def es_sigma(_min, _max):
    return  (_max-_min)  / 2.0 / 3.1 

def es_gaussian_noise(_min, _max):
    mean = (_max+_min)  / 2.0 
    dev = es_sigma(_min,_max)
    ra = random.gauss(mean, dev)
    while (_min <= ra <= _max) == False:
        ra = random.gauss(mean, dev)
    return ra

class ParamerESOptimizer(object):

    def __init__(self, fitness):  
        self.fitness = fitness      
        self.init_learning_rate = 0.1
        self.decay = 0.75
        self.popsize = 5
        self.ngens = 3
        self.learning_rate = self.init_learning_rate
    
    def _start_individual_from_domain(self):
        genotype = []
        for i, d in enumerate(self.domains):
            genotype.append(d.value)
        return genotype

    def _gen_offspring(self, n):
        S = []
        P = [[0 for _ in range(n.nparams())] for _ in range(self.popsize)]
        for i in range(self.popsize):
            j = 0
            for f in n.genotype:
                for d in n.filters[f].domains:
                    sigma = es_sigma(*d.domain) * 0.25
                    nparam = n.params[j] + random.gauss(0, sigma)
                    nparam = max(d.domain[0], min(nparam, d.domain[1]))
                    P[i][j] = d.ptype.ptype(nparam)
                    S.append(sigma)
                    j += 1
        return P,S

    @staticmethod
    def _normalize(n):
        j = 0
        for f in n.genotype:
            for d in n.filters[f].domains:
                n.params[j] = max(d.domain[0], min(n.params[j], d.domain[1]))
                n.params[j] = d.ptype.ptype(n.params[j])
                j += 1

    def _evaluate(self, n, X, Y, P):
        R = np.zeros(len(P)) 
        for p in range(len(P)):
            _Xf = np.array([n.apply(X[i], P[p]) for i in range(X.shape[0])])
            R[p] = self.fitness(_Xf, X, Y)    
            del _Xf
        return R

    def _compute_gradient(self, R):
        std = np.std(R)
        if std == 0.0:
            return np.zeros_like(R)
        return (R - np.mean(R)) / std

    def step(self, n, X, Y):
        P,S = self._gen_offspring(n)
        R = self._evaluate(n, X, Y, P)
        A = self._compute_gradient(R)
        L = n.nparams()
        P = np.array(P)
        grad = np.dot(P.T,A)
        if len(grad.shape) and grad.shape[0]:
            for i in range(L):
                n.params[i] += (self.learning_rate / (L*S[i])) * grad[0]
        self.learning_rate *= self.decay
        #delate all np arrays
        del P, S, R, A, L

    def fit(self, n, X, Y):
        self.learning_rate = self.init_learning_rate
        for _ in range(self.ngens):
            self.step(n, X, Y)
        self._normalize(n)
        return n


class PatamerGAOptimizer(object):

    def __init__(self, fitness, compare):  
        self.fitness = fitness     
        self.compare = compare 
        self.popsize = 5
        self.ngens = 3
        self.population = []
        self.newpopulation = []
        self.domains = []
        self.elite = []

    def one_point_crossover(self, x, y):
        index = random.randrange(1,len(x)-1)
        return x[0:index] + y[index:]

    def eval_params(self, n, X, Y, P):
        Xf = np.array([n.apply(X[i], P) for i in range(X.shape[0])])
        return self.fitness(Xf, X, Y)

    def crossover(self):
        if len(self.domains) > 1:
            p = self.population[random.randrange(0, len(self.population))]
            m = self.population[random.randrange(0, len(self.population))]
            y = self.one_point_crossover(p,m)
        else:
            y = self.population[random.randrange(0, len(self.population))]
        return y

    def mutation(self, x):
        for i, d in enumerate(self.domains):
            if random.uniform(0,1) < 0.5:
                x[i] = d.random()
        return x

    def selection(self, n, X, Y, m, i):
        if  self.compare(self.eval_params(n, X, Y, m), self.eval_params(n, X, Y, self.population[i])):
            self.newpopulation[i] = m
        else:
            self.newpopulation[i] = self.population[i]

    def elitism(self, n, X, Y, m):
        if self.compare(self.eval_params(n, X, Y, m), self.eval_params(n, X, Y, self.elite)):
            self.elite = m

    def step(self, n, X, Y, i):
        y = self.crossover()
        m = self.mutation(y)
        self.selection(n, X, Y, m, i)
        self.elitism(n, X, Y, m)

    def fit(self, n, X, Y):
        self.domains = ga_concat([[d for d in n.filters[fid].domains] for fid in n.genotype])
        self.population = [
            ga_concat([[d.random() for d in n.filters[fid].domains] for fid in n.genotype]) for p in range(self.popsize)
        ]
        self.newpopulation = [None for p in range(self.popsize)]
        self.elite = n.params
        for _ in range(self.ngens):
            for i in range(self.popsize):
                self.step(n, X, Y, i)
            self.population = self.newpopulation
            self.newpopulation = [None for p in range(self.popsize)]
        n.params = self.elite
        return n



class AGVOptimizer(object):
    
    @staticmethod
    def mutation(x, domain):
        for i in range(len(x)):
            if random.uniform(0,1) < 0.5:
                x.change(i, random.randint(domain[0],domain[1]), rand_params=True)
        return x

    @staticmethod
    def mutation_random_params(x):
        j = 0
        for f in x.genotype:
            for d in x.filters[f].domains:
                if random.uniform(0,1) < 0.5:
                    x.params[j] = d.random()
                j += 1
        return x

    @staticmethod
    def one_point_crossover(x,y):
        index = random.randrange(1,len(x)-1)
        return x.pice(0,index) + y.pice(index)

    @staticmethod
    def two_point_crossover(x,y):
        index_s = random.randrange(1,len(x)-1)
        index_e = random.randrange(index_s,len(x))
        return x.pice(0,index_s) + y.pice(index_s,index_e) + x.pice(index_e)

    def selection_raking(self, offsprings):
        #select
        all_elements = self.population + offsprings
        all_elements = sorted(all_elements, key=lambda x: x.fitness)
        self.population = all_elements[0:len(self.population)]

    def selection_pareto(self, offsprings):
        #select
        all_elements = [offspring for offspring in offsprings]
        all_elements+= [parent for parent in self.population]
        to_select = [] # new list containing elements that don't lead to a class change
        for element in all_elements: # find these elements checking the inv_attack_rate
            if inv_attack_rate(self.model, np.expand_dims(element.X, axis=0), np.expand_dims(element.apply(element.X), axis=0)) == 1:
                to_select.append(element)
        
        self.selection_log.append(20 - len(to_select))#class change logging

        if len(to_select) == 0: # If there are no available filters to use return to next epoch
            return

        new_pop = nsga_2_pass(len(self.population), [e.fitness for e in to_select])
        self.population = [to_select[p] for p in new_pop]

    def __init__(self, 
                 Nf,   
                 filters,
                 NP,
                 fitness,
                 img_id,
                 model_path,
                 fitness_max = float("inf"), 
                 params_strategy = "direct", # or tournament
                 params_optimizer = "ES", # GA or random
                 params_pool = "offsprings", # and/or "|parents"
                 selection_type = "ranking", # or pareto (|no-params)
                 use_elitims = True,
                 repetitions = True,
                 logs_fitness = None,
                 save_candidates = None,
                 logs_path = "stats.txt",
                 save_state = None,
                 X = None
                 ):

        if  (not repetitions) and (len(filters) < Nf):
            raise "For this test number of filters has to be smaller (or equal) than the size of all possible filters"

        self.repetitions = repetitions
        self.population = [Individual(Nf,
                                      filters,  
                                      fitness_max,
                                      self.repetitions,
                                      X) for _ in range(NP)]
        self.filters = filters
        self.fitness = fitness
        self.ga_domain = [0, len(filters)-1]
        self.ga_domain_values = [v for v in range(len(filters))]
        self.params_strategy = params_strategy
        self.params_optimizer = params_optimizer
        self.params_pool = params_pool
        self.use_elitims = use_elitims
        self.logs_fitness = logs_fitness
        self.logs_path = logs_path
        self.save_state = save_state

        self.img_id = img_id
        self.model_path = model_path
        #create file
        if self.logs_fitness is not None:
            P = os.path.splitext(self.model_path)[0]
            mkdir_p(P)
            P = os.path.join(P, "outs")
            mkdir_p(P)
            P = os.path.join(P, "img_"+ str(self.img_id)+ "_" + self.logs_fitness)
            with open(P,"w"):
                pass
        ##
        self._last_epoch = 0
        self._pbest = None
        self._first= True

        # attribute for class change logging
        self.model = build_model()
        self.selection_log = []

        #test        
        if  selection_type.find("pareto") >= 0:
            sq_dis_to_0 = lambda x: (x[0]+x[1])**2 
            #sq_dis_to_0 = lambda x: (x[0])**2 + (x[1])**2 
            self.compare_tournament = lambda f1, f2 : dominates(f1, f2)
            self.compare_elit = lambda f1, f2 : sq_dis_to_0(f1) <=  sq_dis_to_0(f2)
            if selection_type.find("no-params") >= 0:
                #for params
                self.compare_params = lambda f1, f2 : f1 <= f2                 
                self.fitness_params = lambda Xf, X, Y: self.fitness(Xf, X, Y, True)
                self.selection_type = "pareto"
            else:
                #for params
                self.compare_params = lambda f1, f2 : sq_dis_to_0(f1) <=  sq_dis_to_0(f2)
                self.fitness_params = lambda Xf, X, Y: self.fitness(Xf, X, Y)
                if params_optimizer == "ES":            
                    self.fitness_params = lambda Xf, X, Y: sq_dis_to_0(self.fitness(Xf, X, Y))
            #set as pareto
            self.selection_type = "pareto"
        else:
          
            self.compare_tournament = \
            self.compare_params = \
            self.compare_elit = lambda f1, f2 : f1 <= f2 
            #for params
            self.compare_params = lambda f1, f2 : f1 <= f2 
            self.fitness_params = lambda Xf, X, Y: self.fitness(Xf, X, Y)
            #set as ranking
            self.selection_type = "ranking"

    def evaluate(self, n, X, Y):
        Xf = np.array([n.apply(X[i]) for i in range(X.shape[0])])
        return self.fitness( Xf, X, Y )
    
    def evaluate_set(self, set_to_eval, X, Y):       
        for n in set_to_eval:
            n.fitness = self.evaluate(n, X, Y)                 

    def evaluate_population(self, X, Y):
        self.evaluate_set(self.population, X, Y)

    def optimize_params_and_eval(self, n, X, Y):
        if self.params_optimizer == "ES":
            ParamerESOptimizer(self.fitness_params).fit(n, X, Y)
            n.fitness = self.evaluate(n, X, Y)        
            return n
        if self.params_optimizer == "GA":
            PatamerGAOptimizer(self.fitness_params, self.compare_params).fit(n, X, Y)
            n.fitness = self.evaluate(n, X, Y)
            return n
        elif self.params_optimizer == "random":
            AGVOptimizer.mutation_random_params(n)
            n.fitness = self.evaluate(n, X, Y)
            return n
        else:
            n.fitness = self.evaluate(n, X, Y)
            return n

    def apply_params_strategy(self, offsprings, X, Y):
        if self.params_strategy == "direct":
            for i,n in enumerate(offsprings):
                offsprings[i] = self.optimize_params_and_eval(n, X, Y)
        elif self.params_strategy == "tournament":
            for i,n in enumerate(offsprings):
                n.fitness = self.evaluate(n, X, Y)
                n1 = self.optimize_params_and_eval(copy.deepcopy(n), X, Y)
                if self.compare_tournament(n1.fitness, n.fitness):
                    offsprings[i] = n1
                else:
                    offsprings[i] = n
        else: #None
            pass
        return offsprings

    def elitism(self, X, Y):
        if self._pbest is None:
            self._pbest = self.population[0]
        else:            
            self.evaluate(self._pbest, X, Y )
        for p in self.population:
            if self.compare_elit(p.fitness,self._pbest.fitness):
                self._pbest = p 

    def gen_offspring(self):
        #item = 0
        # TODO: per i parametri dei genitori non eseguire ES
        p = self.population[random.randrange(0, len(self.population))]
        m = self.population[random.randrange(0, len(self.population))]
        y = self.one_point_crossover(p, m)
        n = self.mutation(y, self.ga_domain)
        #############
        # TEST ###### #when repetitions= false
        if not self.repetitions:
            #classic version
            for i in range(len(n.genotype)-1,-1,-1): 
                if n.genotype[i] in n.genotype[:i]: 
                    domain = [f for f in self.ga_domain_values if not (f in n.genotype)]
                    n.change(i, random.sample(domain,1)[0], rand_params=True)
        return n

    def fit_pass(self, X, Y):
        #eval first pop
        if self._first:
            self.evaluate_set(self.population, X, Y)
            self._first = False
            
        #start
        offsprings = []
        for i in range(len(self.population)):
            offsprings.append(self.gen_offspring())
        #optimize params
        if self.params_pool.find("parent") >= 0: #or "parents"
            self.population = self.apply_params_strategy(self.population, X, Y)
        if self.params_pool.find("offspring") >= 0: #or "offsprings"
            offsprings = self.apply_params_strategy(offsprings, X, Y)
        else:
            self.evaluate_set(offsprings, X, Y)
        #select
        if self.selection_type == "pareto":
            self.selection_pareto(offsprings)
        elif self.selection_type == "nsgaiii":
            self.selection_nsgaiii_pymoo(offsprings)
        else:
            self.selection_raking(offsprings)
        #select the best
        if self.use_elitims:
            self.elitism(X, Y)
    
    def _return_best(self):
        #return best
        if self.use_elitims:
            return self._pbest
        else:
            if self.selection_type == "pareto":
                return self.population[int(len(self.population)/2)]
            else:
                return self.population[0]

    def _save_state(self, epoch):
        save_path, model_name, params_to_save = self.save_state
        if save_path is not None:
            with open("{}.{}.{}".format(save_path,epoch,STATE_EXT), "wb") as file_save_state:
                #kersa/tensoflow cannot be serilized
                self.fitness, t_fitness = None, self.fitness
                #list of stuff to save
                serialize = {
                    "optimizer_state" : self,
                    "py_random_state" : random.getstate(),
                    "np_random_state" : np.random.get_state(),
                    "epoch" : epoch
                }
                serialize_and_deserialize.dump(serialize,file_save_state)
                # reset fitness
                self.fitness = t_fitness
            #save best solution
            ModelLoader().from_individual(self._return_best(),{ 
                "neural_network_model" : model_name,
                "in_params" : params_to_save,
            }).save("{}.{}.{}".format(save_path, epoch, JSON_EXT))

    @staticmethod
    def load_state(state_path, fitness):
        deserialize = {}
        with open(state_path, 'rb') as state_file:
            deserialize = serialize_and_deserialize.load(state_file)
        random.setstate(deserialize["py_random_state"])
        np.random.set_state(deserialize["np_random_state"])
        optimizer_state = deserialize["optimizer_state"]
        optimizer_state.fitness = fitness
        return optimizer_state
        
    def fit(self, X, Y, batch, target, target_id, epoch = 5):
        #salvataggio cam per calcolarla solo una volta, da migliorare
        cam_original_image = get_cam(target[0])
        P = str(os.getcwd()) + "/img_cam/img_cam.png"
        plt.imsave(P, cam_original_image, cmap='gray')

        self._first = self._last_epoch == 0
        P = os.path.splitext(self.model_path)[0]
        P = os.path.join(P, "logs_txts")
        mkdir_p(P)
        P = os.path.join(P, "img_"+ str(self.img_id)+ "_" + self.logs_path)
        logs = Log(P, append = not self._first)
        for e in range(self._last_epoch, epoch):
            X, Y = sk_shuffle(X, Y)            
            for i in tqdm(range(int(X.shape[0] / batch)), desc = "epoch {}/{} ".format(e+1,epoch)):
                s_i = i * batch
                e_i = (i+1) * batch
                batch_X, batch_Y = X[s_i:e_i], Y[s_i:e_i]                
                self.fit_pass(batch_X, batch_Y)
                logs.log("{}\t{}\t".format(e,i) + "\t".join(["{},{},{}".format(str(p.genotype),str(p.params),str(p.fitness)) for p in self.population]))
                del batch_X, batch_Y           

            if self.logs_fitness is not None:
                P = os.path.splitext(self.model_path)[0]                
                P = os.path.join(P, "outs")
                P = os.path.join(P, "img_"+ str(self.img_id)+ "_" + self.logs_fitness)
                with open(P,"a+") as fitsfile:
                    best = self._pbest if self.use_elitims \
                    else self.population[int(len(self.population)/2)] if self.selection_type == "pareto" \
                    else self.population[0]

                    Xf = np.array([best.apply(X[i]) for i in range(X.shape[0])])
                    fits = self.fitness(Xf, X, Y)

                    if type(fits) == float: #it is attack rate
                        fitsfile.write("{}\n".format(1.0-fits))
                    else:
                        fitsfile.write("target id: {}\t".format(target_id))
                        fitsfile.write("{}\t{}\n".format(fits[0],fits[1]))
                        # before was fitsfile.write("{}\t{}\n".format(1.0-fits[0],fits[1]))
                
            #update last epoch
            self._last_epoch = e+1
            #save state if needed
            #self._save_state(e)           

        #logging number of elements that caused class change
        with open("TEST/logs_txts/log_selection.txt", "a") as file:
            file.write(f'image id: {self.img_id} --- {self.selection_log}\n')

        return self._return_best()