import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import errno
import pathlib
import random
import json
import numpy as np
from PIL import Image
from PIL import ImageDraw
from filters import show_image
from filters import _to_pil_image
from agv_model_loader import ModelLoader
from agv_datasets import build_model_and_dataset
from agv_metrics import compute_metricts
import pandas as pd
import tensorflow as tf



import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_Cam import *
'''
###
import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM,HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise
    

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget



import matplotlib.pyplot as plt
###
'''

# load classes json data
json_file = open(os.path.join(os.path.join(pathlib.Path(__file__).parent.absolute(), 
                                      "../datasets/imagenet_class_index.json")))
class_names = json.load(json_file)

def one_sample_batch(image):
    batch = np.zeros(shape=(1,*image.shape))
    batch[0,:,:,:]=image
    return batch

def predict_image(model,image):
    return np.argmax(model.predict(one_sample_batch(image))[0])

def predict_image_detailed(model,image):
    return model.predict(one_sample_batch(image))[0]

def dataset_from_model(path_model):
    import json 
    with open(path_model,"r") as jfile:
        return json.load(jfile)["in_params"]["dataset"]

def test_fits(path_model, dataset_name = None):
    if dataset_name is None:
        dataset_name = dataset_from_model(path_model)
    nn_model, X, Y = build_model_and_dataset(dataset_name)
    agv_model = ModelLoader().load(path_model).to_individual()
    #compute metricts
    metricts = compute_metricts(agv_model, nn_model, dataset_name, X, Y)
    #print
    print("Best fit:", agv_model.fitness)
    print("Best genotype:", agv_model.genotype)
    print("Best params:", agv_model.params)
    for key,val in metricts.items():
        print("{}:".format(key), val)

def test(path_model, show_info = False, dataset_name = None):
    if dataset_name is None:
        dataset_name = dataset_from_model(path_model)
    nn_model, X, Y = build_model_and_dataset(dataset_name)
    I = random.randrange(0,X.shape[0])
    model = ModelLoader().load(path_model)
    blank_image = Image.new("RGB",(X.shape[1]*2+3,X.shape[2]+2))
    blank_image.paste(_to_pil_image(X[I]),(1,1)) 
    blank_image.paste(_to_pil_image(model.apply(X[I])),(X.shape[1]+2,1)) 
    print("predict p1 in test function:")
    p1 = predict_image(nn_model, X[I])
    print("predict p2 in test function:")
    p2 = predict_image(nn_model, model.apply(X[I]))
    draw = ImageDraw.Draw(blank_image)
    draw.text((0, 0), "{}".format(p1), (255, 0, 0))
    draw.text((X.shape[1]+1, 0), "{}".format(p2), (255, 0, 0))
    blank_image.show()
    if show_info:
        gf = model.get_filters()
        filters = str([str(fp[0].name)+str(fp[1:]) for fp in gf])
        print(str(filters))
    
def mkdir_p(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def save_adv_ex(path_model, nimages = 10, dataset_name = None):
    if dataset_name is None:
        dataset_name = dataset_from_model(path_model)
    nn_model, X, Y = build_model_and_dataset(dataset_name)
    print(dataset_name, "size:", X.shape[0])
    model = ModelLoader().load(path_model)
    P = os.path.splitext(path_model)[0]
    mkdir_p(P)
    P_t = os.path.join(P,"img{}_{}_{}.png")
    N = 0
    for I in range(X.shape[0]):
        
        p1 = predict_image(nn_model, X[I])
        print("predict p1 in save_adv_ex function:", p1)
        
        p2 = predict_image(nn_model, model.apply(X[I]))
        print("predict p2 in save_adv_ex function:", p2)
        if p1 != p2:
            blank_image = Image.new("RGB",(X.shape[1]*2+3,X.shape[2]+2))
            blank_image.paste(_to_pil_image(X[I]),(1,1)) 
            blank_image.paste(_to_pil_image(model.apply(X[I])),(X.shape[1]+2,1)) 
            draw = ImageDraw.Draw(blank_image)
            draw.text((0, 0), "{}".format(p1), (255, 0, 0))
            draw.text((X.shape[1]+1, 0), "{}".format(p2), (255, 0, 0))
            blank_image.save(P_t.format(I,p1,p2))
            N += 1
            if nimages <= N: 
                return 0
  
def save_adv_best(best_folder, image_id=0, dataset_name = None ):

    best_files = os.listdir(best_folder)
    best_files.sort(key=lambda s: int(s.split("_")[1]))
    #TO DO : condition when there isnt a best json for that image id
    print("best_files:", best_files)
    if dataset_name is None:
      dataset_name = dataset_from_model(os.path.join(best_folder, best_files[image_id]))

    nn_model, X, Y = build_model_and_dataset(dataset_name)
    best_model =  ModelLoader().load(os.path.join(best_folder, best_files[image_id]))
    print("Starting saving best" )
    print(dataset_name, "size:", X.shape[0], "network name: ", nn_model.name)  

    P = (os.path.splitext(best_folder)[0]).split("/")[-2]  
    mkdir_p(P)
    P = os.path.join(P, "generated_best_images")
    mkdir_p(P) 
    P_t = os.path.join(P,"best_img{}_.png")
    blank_image = Image.new("RGB",(224,224))
    blank_image.paste(_to_pil_image(best_model.apply(X[image_id]))) 
    draw = ImageDraw.Draw(blank_image)                
    blank_image.save(P_t.format(image_id))

    p1 = predict_image(nn_model, X[image_id])
    p2 = predict_image(nn_model, best_model.apply(X[image_id]))

    P_tt = os.path.join(P,"OG_vs_best_img{}_{}_{}.png")
    blank_image = Image.new("RGB",(X.shape[1]*2+3,X.shape[2]+2))
    blank_image.paste(_to_pil_image(X[image_id]),(1,1)) 
    blank_image.paste(_to_pil_image(best_model.apply(X[image_id])),(X.shape[1]+2,1)) 
    draw = ImageDraw.Draw(blank_image)
    draw.text((0, 0), "{}".format(p1), (255, 0, 0))
    draw.text((X.shape[1]+1, 0), "{}".format(p2), (255, 0, 0))
    blank_image.save(P_tt.format(image_id,p1,p2))
    
    modified_image = cv2.imread(P_t.format(image_id), 1)    #prendo immagine creata con filtro applicato
    model = models.resnet50(pretrained=True)
    target_layers = [model.layer4]
    original_image = np.float32(X[image_id])
    input_tensor = preprocess_image(original_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    cam_algorithm = GradCAM

    with cam_algorithm(model = model, target_layers = target_layers) as cam:
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]
        mask = grayscale_cam * 255  #make range between 0-255
    
    retval, img_thresh = cv2.threshold(mask, 155, 255, cv2.THRESH_BINARY) #threshold alla maschera per filtrare la zona focale
    mask = img_thresh.astype(np.uint8)

    img_applied_mask = cv2.bitwise_and(modified_image, modified_image, mask = mask)   #seziono l'immagine modificata con la maschera
    #plt.imshow(img_applied_mask[:, :, ::-1])
    #plt.show()

    #filtered_img = cv2.imread(P_t.format(image_id), 1)
    #filtered_img = cv2.bitwise_and(filtered_img, filtered_img, mask = mask)

    img_logo_mask_inv = cv2.bitwise_not(mask)
    img_foreground = cv2.bitwise_and(original_image, original_image, mask = img_logo_mask_inv) *255
    img_foreground = img_foreground.astype(np.uint8)

    result = cv2.add(img_applied_mask, img_foreground[:, :, ::-1])
    cv2.imwrite(P+'/gradcam.jpeg', result)










