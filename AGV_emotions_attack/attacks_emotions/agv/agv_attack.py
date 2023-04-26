import sys, os
import random
import argparse
#set modules path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(os.path.join(os.path.dirname(__file__),'..', ".."))
import numpy as np
import pandas as pd

from fitness import inv_attack_rate, inv_attack_rate_multiple
from agv_filters import Filters
from agv_model_loader import ModelLoader
from agv_optimizer import AGVOptimizer
from agv_optimizer import Individual
from agv_datasets import build_model_and_dataset
from agv_datasets import get_model_name_from_dataset
from agv_datasets import database_and_model
from agv_distances import get_distance_functions
from agv_metrics import compute_metricts

from agv_tests import *
#from agv_tests import test, test_fits, save_adv_ex

from agv_tests import mkdir_p, save_adv_best
from log import Log

import time
start_time = time.time()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)

P = os.getcwd()
P = str(P) + '/img_cam/'
from random import choice #new method
if not os.path.exists(P):
  # if the demo_folder directory is not present 
  # then create it.
  os.makedirs(P)

def main(dataset_name,
         model_path, 
         logs_path,
         training_state,
         number_of_filters,
         population_size,
         batch_size, 
         epochs, 
         params_strategy, 
         params_optimizer, 
         params_pool,
         use_elitims,
         repetitions,
         distance_function_one = "norm_1",
         selection="ranking",
         logs_fitness = None,         
         params_to_save = {}):

    #block seed
    random.seed(42)
    
    #get model and dataset
    model_one, X, Y = build_model_and_dataset(dataset_name)
    print("shape of X: ", X.shape)

    #get model name 
    model_name = get_model_name_from_dataset(dataset_name)
    print("the models are: ", model_name)

    #fitness 
    #f_quality  = lambda Xf, X : float(inv_attack_rate(model_one, Xf, X ))

    f_distance_xai = get_distance_functions(dataset_name, model_one)['ssim_score_target']
    f_distance_one = get_distance_functions(dataset_name, model_one)[distance_function_one]
    #by ranking or by pareto
    if selection == "ranking":
        fit = lambda Xf, X, Y  : f_distance_xai(Xf, X) #+ f_distance(Xf,X) * 0.95     
    else: #pareto or pareto|no-params
      # if selection == "pareto":
        fit = lambda Xf, X, Y, only_quality = False: \
                [f_distance_xai(Xf, X), f_distance_one(Xf, X)] if not only_quality else f_distance_xai(Xf, X)

    #train AGV 
    opt = None
    #init opt
    if os.path.exists(training_state):
        opt = AGVOptimizer.load_state(training_state, fitness=fit)
        #hack
        if logs_fitness is not None:
            opt.logs_fitness = logs_fitness
        if logs_path is not None:
            opt.logs_path = logs_path
    else:  
        
        best = []         
        prediction_on_bestind = {}
        for i in range(X.shape[0]):
          print("Exec optimizer on image :", i)  
          opt = AGVOptimizer(Nf=number_of_filters, 
                              filters=Filters,
                              NP=population_size, 
                              fitness=fit,
                              img_id = i,
                              model_path = model_path,
                              fitness_max=float("inf"),                        
                              params_strategy=params_strategy,                        
                              params_optimizer=params_optimizer,                        
                              params_pool=params_pool,
                              selection_type=selection,
                              use_elitims=use_elitims,
                              repetitions=repetitions,
                              logs_fitness=logs_fitness,                              
                              logs_path = logs_path,
                              save_state=(training_state, model_name, params_to_save),
                              X = X[i]
                            )
          OG_class = np.argmax(model_one.predict(X[i:i+1]))
          print(X[i:i+1].shape, X[i:i+1].mean(), "original class:", Y[i:i+1], np.argmax(Y[i:i+1]), ", predicted class:", OG_class)
          # print(f'X SHAPE {X.shape[0]}')
          target_id = choice([target_choice for target_choice in range(X.shape[0]) if target_choice not in [i]])
          print(f'CHOICE {target_id}')
          best_ind = opt.fit(X[i:i+1], Y[i:i+1], batch_size, X[target_id:target_id+1], target_id, epochs)
          
          best.append(best_ind)
          print("class of original image: ", OG_class )
          X_modified_with_best = np.array(best_ind.apply(X[i]))
          X_modified_with_best =  np.expand_dims(X_modified_with_best, axis=0)
          
          MOD_class = np.argmax(model_one.predict(X_modified_with_best))
          print("class of modified image: ", MOD_class)

          prediction_on_bestind[i] = {"original_class":OG_class,
                                      "modified class":MOD_class}
          #compute all fit
          print("Getting info:")           
          print("Best fit:", best_ind.fitness)
          print("Best genotype:", best_ind.genotype)
          print("Best params:", best_ind.params)
    
          #save
          P = os.path.splitext(model_path)[0]  
          P = os.path.join(P, "best_jsons")
          mkdir_p(P)
          P_t = os.path.join(P,"img_" +  str(i)+ "_" + model_path)
          ModelLoader().from_individual(best_ind,{               
              "neural_network_model" : model_name,
              "in_params" : params_to_save,
          }).save(P_t)
          
        #saving best list to a file, just for future reference:
        class_info_df = pd.DataFrame.from_dict(prediction_on_bestind, orient='index').reset_index() 
        # P_df = os.path.join(P,  "class_info_df.csv")
        # class_info_df.to_csv(P_df, encoding='utf-8', index=False)
        for i,ind in enumerate(best):
          print(i, ind.fitness, ind.genotype, ind.params)      
        P = os.path.splitext(model_path)[0]
        P = os.path.join(P, "final_outcome") 
        mkdir_p(P)    
        P = os.path.join(P, "final_outcome.txt")      
        logs = Log(P, append = not (i==0))
        for b in best:
          logs.log("\t".join("{},{},{}".format(str(b.fitness),str(b.genotype),str(b.params))))
        
        P = os.path.splitext(model_path)[0]
        P = os.path.join(P, "final_outcome") 
        mkdir_p(P)   
        P_df = os.path.join(P,  "class_info_df.csv")
        class_info_df.to_csv(P_df, encoding='utf-8', index=False)
  
    print("--- %s seconds ---" % (time.time() - start_time))

                    


def str2bool(v):
    return v if isinstance(v, bool) else \
           True if v.lower() in ('yes', 'true', 't', 'y', '1') else \
           False

def args_to_dict(arguments):
    return { key :  str(arguments.__dict__[key]) for key in  arguments.__dict__ }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset","-db", choices=list(database_and_model().keys()), default=None)
    parser.add_argument("--logs","-l", type=str,default=None)
    parser.add_argument("--logs_fitness","-lf", type=str,default=None)
    parser.add_argument("--input","-i", type=str,default=None)
    parser.add_argument("--test","-t", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--repetitions","-r", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--save_adv_ex","-sae", type=int, nargs='?', default=0)
    parser.add_argument("--show_info","-si", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--number_of_filters","-nf", type=int, default=3)
    parser.add_argument("--population_size","-np", type=int, default=6)
    parser.add_argument("--batch_size","-bs", type=int, default=100)
    parser.add_argument("--epochs","-e", type=int, default=1)
    parser.add_argument("--training_state","-ts", type=str, default="")
    parser.add_argument("--params_optimizer","-po", choices=["ES", "GA", "DE","smac", "random", "none"], default="ES",
                        help="select the type of the parameter optimizer")
    parser.add_argument("--params_strategy","-ps", choices=["direct","tournament","none"],default="direct",
                       help="select the strategy kinda used to choose the offspring's parameters")
    parser.add_argument("--params_pool","-pp", type=str, default="offsprings", 
                       help="select which is the population pool where to apply the parameter optimization (offsprings|parents)")
    parser.add_argument("--selection","-s", choices=["ranking","pareto","pareto|no-params", "nsgaiii"],default="ranking",
                        help="select the type of the selection strategy")
    parser.add_argument("--elitims","-el",  type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--distance_function_one","-df1",  choices=list(get_distance_functions().keys()), default="norm_1",
                        help="select the distance function")   
    parser.add_argument("--output","-o", type=str,default=None)
    parser.add_argument("--best_folder","-bf", type=str,default=None)
    parser.add_argument("--save_adv_best","-sae_best", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--image_id","-img_id", type=int, nargs='?', default=0)
    inargs = parser.parse_args()
    if type(inargs.output) is str:
        main(inargs.dataset,             
             inargs.output, 
             inargs.logs, 
             inargs.training_state,
             inargs.number_of_filters, 
             inargs.population_size, 
             inargs.batch_size, 
             inargs.epochs, 
             inargs.params_strategy, 
             inargs.params_optimizer, 
             inargs.params_pool,
             inargs.elitims,
             inargs.repetitions,
             inargs.distance_function_one,
             inargs.selection,
             inargs.logs_fitness,             
             params_to_save=args_to_dict(inargs))
     
    if inargs.save_adv_ex > 0:
        save_adv_ex(inargs.input if type(inargs.input) is str else inargs.output,
                    nimages = inargs.save_adv_ex,
                    dataset_name=inargs.dataset)
    if type(inargs.best_folder) is str and (inargs.save_adv_best) and inargs.image_id >= 0:
        save_adv_best(inargs.best_folder, image_id = inargs.image_id, dataset_name=inargs.dataset )
    #----
