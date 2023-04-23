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

def get_iou(ground_truth, pred):
	"""
    Calculate Intersection over Union of 2 bounding rectangle.
    Arguments:
        ground_truth: an array of dim 4 --> [x1, y1, x2, y2]
			where (x1, y1) and (x2, y2) are the top-left and bottom-right
			corner of the bounding rect
        pred: an array of dim 4 --> [x1, y1, x2, y2]
			where (x1, y1) and (x2, y2) are the top-left and bottom-right
			corner of the bounding rect
    Returns:
        The IoU of the 2 bounding rect.
	"""
	# coordinates of the area of intersection.
	ix1 = np.maximum(ground_truth[0], pred[0])
	iy1 = np.maximum(ground_truth[1], pred[1])
	ix2 = np.minimum(ground_truth[2], pred[2])
	iy2 = np.minimum(ground_truth[3], pred[3])

	# Intersection height and width.
	i_height = np.maximum(iy2 - iy1, np.array(0.))
	i_width = np.maximum(ix2 - ix1, np.array(0.))
	area_of_intersection = i_height * i_width

	# Ground Truth dimensions.
	gt_height = ground_truth[3] - ground_truth[1]
	gt_width = ground_truth[2] - ground_truth[0]

	# Prediction dimensions.
	pd_height = pred[3] - pred[1]
	pd_width = pred[2] - pred[0]

	area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
	iou = area_of_intersection / area_of_union
	return iou

def get_center_distance(ground_truth, pred):
    """
    Calculate Distance between Center of 2 bounding rectangle.
    Arguments:
        ground_truth: an array of dim 4 --> [x1, y1, x2, y2]
			where (x1, y1) and (x2, y2) are the top-left and bottom-right
			corner of the bounding rect
        pred: an array of dim 4 --> [x1, y1, x2, y2]
			where (x1, y1) and (x2, y2) are the top-left and bottom-right
			corner of the bounding rect
    Returns:
        The Distance between Center of the 2 bounding rect.
    """
    # Center Point of ground_truth
    cx1 = (ground_truth[0] + ground_truth[2]) / 2
    cy1 = (ground_truth[1] + ground_truth[3]) / 2
    # Center Point of Attacked image
    cx2 = (pred[0] + pred[2]) / 2
    cy2 = (pred[1] + pred[3]) / 2
    # Computing distance
    distance = math.dist([cx1, cy1], [cx2, cy2])
    #min-max rescaling
    return distance / 315.36

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
    SSIM +=  ssim(X2[sample],X1[sample],data_range = 1, channel_axis=-1)
  return 1 - (SSIM / X1.shape[0])#if the two images are identical then the returned score will be zero; 1 otherwise


def ssim_score_not_inv(X1, X2):
    """
    ssim to use with the explaination cam. It is not inverted beacuse we want to minimize it
    X1 is the Xf, X2 is the original
    """
    thresh_yes = False
    X2 = cv2.imread('img_cam/img_cam.png',cv2.IMREAD_GRAYSCALE)
    if thresh_yes:
        X2 = np.uint8(X2)
        threshold_value = 200
    else:
        X2 = np.float32(X2) / 255

    # original_image = np.float32(X2[0])
    # input_tensor_original = preprocess_image(original_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    modified_image = np.float32(X1[0])
    input_tensor_modified = preprocess_image(modified_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    cam_algorithm = EigenCAM
    with cam_algorithm(model = MODEL_gradcam, target_layers = TARGET_LAYERS, use_cuda = True) as cam:
        cam.batch_size = 256
        # grayscale_cam_eigen_original = cam(input_tensor=input_tensor_original, targets=None)
        grayscale_cam_eigen_modified = cam(input_tensor=input_tensor_modified, targets=None)
        # X2 = grayscale_cam_eigen_original[0, :]
        X1 = grayscale_cam_eigen_modified[0, :]

    
    if thresh_yes:
        modified_image_cam = np.uint8(X1 * 255)
        _, thresh_original_image = cv2.threshold(X2, threshold_value, 255, cv2.THRESH_TOZERO)
        _, thresh_modified_image = cv2.threshold(modified_image_cam, threshold_value, 255, cv2.THRESH_TOZERO)
        X2 = np.float32(thresh_original_image) / 255
        X1 = np.float32(thresh_modified_image) / 255

    SSIM = ssim(X2, X1, data_range = 1, channel_axis=-1)
    return SSIM


def IoU(X1, X2):
    """
    X1 is the Xf, X2 is the original
    """
    original_image_cam = cv2.imread('/XAI_AML/AGV-Project/AGV_emotions_attack/img_cam/img_cam.png',cv2.IMREAD_GRAYSCALE)

    modified_image = np.float32(X1[0])
    input_tensor_modified = preprocess_image(modified_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    cam_algorithm = EigenCAM
    with cam_algorithm(model = MODEL_gradcam, target_layers = TARGET_LAYERS, use_cuda = True) as cam:
        cam.batch_size = 128
        grayscale_cam_eigen_modified = cam(input_tensor=input_tensor_modified, targets=None)
        X1 = grayscale_cam_eigen_modified[0, :]
    
    modified_image_cam = np.uint8(X1 * 255)
    _, thresh_original_image = cv2.threshold(original_image_cam, 170, 255, cv2.THRESH_BINARY)
    _, thresh_modified_image = cv2.threshold(modified_image_cam, 170, 255, cv2.THRESH_BINARY)
    contours_original_image, _ = cv2.findContours(thresh_original_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_modified_image, _ = cv2.findContours(thresh_modified_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x_o,y_o,w_o,h_o = cv2.boundingRect(contours_original_image[0])
    x_m,y_m,w_m,h_m = cv2.boundingRect(contours_modified_image[0])
    return get_iou([x_o, y_o, x_o + w_o,  y_o + h_o], [x_m, y_m, x_m + w_m, y_m + h_m])


def center_distance(X1, X2):
    """
    X1 is the Xf, X2 is the original
    """
    threshold_value = 170
    original_image_cam = cv2.imread('img_cam/img_cam.png',cv2.IMREAD_GRAYSCALE)
    modified_image = np.float32(X1[0])
    input_tensor_modified = preprocess_image(modified_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    cam_algorithm = EigenCAM
    with cam_algorithm(model = MODEL_gradcam, target_layers = TARGET_LAYERS, use_cuda = True) as cam:
        cam.batch_size = 128
        grayscale_cam_eigen_modified = cam(input_tensor=input_tensor_modified, targets=None)
        X1 = grayscale_cam_eigen_modified[0, :]
    
    modified_image_cam = np.uint8(X1 * 255)
    _, thresh_original_image = cv2.threshold(original_image_cam, threshold_value, 255, cv2.THRESH_BINARY)
    _, thresh_modified_image = cv2.threshold(modified_image_cam, threshold_value, 255, cv2.THRESH_BINARY)
    contours_original_image, _ = cv2.findContours(thresh_original_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_modified_image, _ = cv2.findContours(thresh_modified_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x_o,y_o,w_o,h_o = cv2.boundingRect(contours_original_image[0])
    x_m,y_m,w_m,h_m = cv2.boundingRect(contours_modified_image[0])
    # Inverting the center distance because we want to maximize it
    return 1 - get_center_distance([x_o, y_o, x_o + w_o,  y_o + h_o], [x_m, y_m, x_m + w_m, y_m + h_m])


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