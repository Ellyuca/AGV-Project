#IL SEGUENTE FILE DEVE ESSERE RIPOSIZIONATO NELLA CARTELLA AGV PER POTER FUNZIONARE CORRETTAMENTE
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

from agv_tests import test, test_fits, save_adv_ex  
from agv_tests import mkdir_p, save_adv_best
from log import Log

from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
import tensorflow as tf

model_one, X, Y = build_model_and_dataset("IMAGENET-MOBILENET")
predictions1 = model_one.predict(X[0:1])
print(imagenet_utils.decode_predictions(predictions1))

img_path = '../datasets/images_dataset/ILSVRC2012_img_val_labeled_caffe_200/65.1.JPEG'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array_expanded_dims = np.expand_dims(img_array, axis=0)
new_img = tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
model = tf.keras.applications.MobileNet()
predictions2 = model.predict(new_img)

print(imagenet_utils.decode_predictions(predictions2))

print()
#print(predictions1-predictions2)