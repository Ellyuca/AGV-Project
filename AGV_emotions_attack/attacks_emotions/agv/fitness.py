import sys, os
import math
import random
import numpy as np
import multiprocessing
import tensorflow as tf 
import tensorflow.keras as K
from scipy.stats import entropy
from functools import reduce
import cv2

from skimage.metrics import structural_similarity as ssim
from agv_xai_utils import *

#set modules path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(os.path.join(os.path.dirname(__file__),'..', ".."))

fitness_manager = multiprocessing.Manager()
fitness_mutex = fitness_manager.Lock()

def _predict(model,X):
    return model.predict(X)

def attack_rate_vs_y(model, X, Y):
    count = 0
    Ypred = _predict(model,X)
    for sample in range(X.shape[0]):
        count += np.argmax(Ypred[sample]) != np.argmax(Y[sample])
    return float(count) / X.shape[0]

def mean_norm_0(X1, X2):
    img_size = X1.shape[1] * X1.shape[2]
    nb_channels = X1.shape[3]

    mean_l0_dist_value = np.mean([ np.sum(X1[i]-X2[i] != 0) for i in range(len(X1))])
    mean_l0_dist_value = mean_l0_dist_value / (img_size*nb_channels)

    diff_channel_list = np.split(X1-X2 != 0, nb_channels, axis=3)
    l0_channel_dependent_list = np.sum(reduce(lambda x,y: x|y, diff_channel_list), axis = (1,2,3))
    mean_l0_dist_pixel = np.mean(l0_channel_dependent_list) / img_size
    return mean_l0_dist_pixel

def mean_norm_1(X1, X2):
    norm_1_acc = 0.0
    for sample in range(X1.shape[0]):
        norm_1_acc += cv2.norm(X1[sample]-X2[sample],  cv2.NORM_L1)
    return norm_1_acc / X1.shape[0] #AVG among all samples

def mean_norm_2(X1, X2):
    norm_2_acc = 0.0
    for sample in range(X1.shape[0]):
        norm_2_acc += cv2.norm(X1[sample]-X2[sample],  cv2.NORM_L2)
    return norm_2_acc / X1.shape[0] #AVG among all samples

def mean_norm_inf(X1, X2):
    norm_inf_acc = 0.0
    for sample in range(X1.shape[0]):
        norm_inf_acc += cv2.norm(X1[sample]-X2[sample],  cv2.NORM_INF)
    return norm_inf_acc / X1.shape[0] #AVG among all samples


def ssim_score(X1,X2): #X1 is the Xf, X2 is the original; order shoudlnt matter though, but just for future reference
  #implementation working for single images only.
  #to do: adapt it for multi image approach, like the norm metrics above
  # X1 = np.squeeze(X1, axis=0)
  # X2 = np.squeeze(X2, axis=0)
  # score = ssim(X2,X1,data_range = 1, multichannel=True)
  SSIM = 0.0
  for sample in range(X1.shape[0]):     
    SSIM +=  ssim(X2[sample],X1[sample],data_range = 1, multichannel=True)
  return 1 - (SSIM / X1.shape[0])#if the two images are identical then the returned score will be zero; 1 otherwise


def ssim_score_not_inv(X1, X2): #ssim to use with the explaination cam. It is not inverted beacuse we want to minimize it
    #X1 is the Xf, X2 is the original
    X2 = cv2.imread('/XAI_AML/AGV-Project/AGV_emotions_attack/img_cam/img_cam.png',cv2.IMREAD_GRAYSCALE)

    # original_image = np.float32(X2[0])
    modified_image = np.float32(X1[0])
    # input_tensor_original = preprocess_image(original_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor_modified = preprocess_image(modified_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    cam_algorithm = EigenCAM
    with cam_algorithm(model = MODEL_gradcam, target_layers = TARGET_LAYERS, use_cuda = True) as cam:
        cam.batch_size = 128
        # grayscale_cam_eigen_original = cam(input_tensor=input_tensor_original, targets=None)
        grayscale_cam_eigen_modified = cam(input_tensor=input_tensor_modified, targets=None)

        # Here grayscale_cam has only one image in the batch
        # X2 = grayscale_cam_eigen_original[0, :]
        X1 = grayscale_cam_eigen_modified[0, :]

    SSIM = ssim(X2, X1, data_range = 1, multichannel=False)
    return SSIM

  
# Note: KL-divergence is not symentric.
# Designed for probability distribution (e.g. softmax output).
def kl(x1, x2):
    assert x1.shape == x2.shape
    # x1_2d, x2_2d = reshape_2d(x1), reshape_2d(x2)

    # Transpose to [?, #num_examples]
    x1_2d_t = x1.transpose()
    x2_2d_t = x2.transpose()

    # pdb.set_trace()
    e = entropy(x1_2d_t, x2_2d_t)
    e[np.where(e==np.inf)] = 2
    return e

def attack_rate(model, Xf, X):
    count = 0
    Yf_pred = _predict(model,Xf)    
    Y_pred =  _predict(model,X)    
    for sample in range(X.shape[0]):    
        count += np.argmax(Yf_pred[sample]) != np.argmax(Y_pred[sample])
        #print("prediction on original image:", np.argmax(Y_pred[sample]))
        #print("prediction on modified image:", np.argmax(Yf_pred[sample]))
    return float(count) / X.shape[0]

def inv_attack_rate(model, Xf, X):
    #print(1.0 - attack_rate(model, Xf, X))
    return 1.0 - attack_rate(model, Xf, X)


# def inv_attack_rate_multiple(model_one, model_two, model_three, model_four, Xf, X, Xf_inc, X_inc):
#     #print(1.0 - attack_rate(model, Xf, X))
#     attack_model_one = attack_rate(model_one, Xf, X)
#     attack_model_two = attack_rate(model_two, Xf, X)
#     attack_model_three = attack_rate(model_three, Xf, X)
#     attack_model_four = attack_rate(model_four, Xf_inc, X_inc)
#     print("attacks on various models:", attack_model_one, attack_model_two, attack_model_three, attack_model_four)
#     attacks_count = attack_model_one + attack_model_two + attack_model_three + attack_model_four
#     norm_attacks_coount = attacks_count/4
#     #return 1.0 - attack_rate(model, Xf, X)
#     return 1.0 - norm_attacks_coount

def inv_attack_rate_multiple(model_one, model_two, model_three, Xf, X):
    #print(1.0 - attack_rate(model, Xf, X))
    attack_model_one = attack_rate(model_one, Xf, X)
    attack_model_two = attack_rate(model_two, Xf, X)
    attack_model_three = attack_rate(model_three, Xf, X)  
    print("attacks on various models:", attack_model_one, attack_model_two, attack_model_three)
    attacks_count = attack_model_one + attack_model_two + attack_model_three 
    norm_attacks_coount = attacks_count/3
    #return 1.0 - attack_rate(model, Xf, X)
    return 1.0 - norm_attacks_coount

def distance_from_eqprob(model, Xf):
    dist = 0.0
    Yf_pred = _predict(model,Xf)
    Yeq = [0.5]*Yf_pred.shape[1]
    for sample in range(Xf.shape[0]):
        mi,ma = np.min( Yf_pred[sample] ) , np.max( Yf_pred[sample] )
        Ynorm = (Yf_pred[sample]-mi) / (ma-mi)
        Ydiff = Yeq - Ynorm
        Ydis  =  np.linalg.norm( Ydiff )
        dist += Ydis
    return dist / Xf.shape[0]

    



class AutoNormalization:
    def __init__(self, max_value = 0.0):
        self.max_value = max_value
    
    def __call__(self, value):
        if value > self.max_value:
            self.max_value =  value
        return value / self.max_value