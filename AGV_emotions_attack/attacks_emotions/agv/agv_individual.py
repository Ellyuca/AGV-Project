import copy
import random


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



class Individual(object):

    def __init__(self, 
                 Nf,
                 filters, 
                 fitness_max,
                 repetitions = True):
        if repetitions:           
            self.genotype = [random.randrange(0, len(filters)) for _ in range(Nf)]        
        else:
            self.genotype = random.sample(range(0, len(filters)), Nf) #
        self.params = []
        self.filters = filters
        for fid in self.genotype:
            self.params += [d.value for d in self.filters[fid].domains]
        self.fitness_max = fitness_max
        self.fitness = fitness_max
    
    def apply(self, image, params = None):
        if params is None:
            params = self.params
        ilast = 0
        for fid in self.genotype:
            ifilter = self.filters[fid]
            '''
            ###
            plt.imshow(image)
            plt.show()
            modello = models.resnet50(pretrained=True)
            #provo a modificare solo la parte con gradcam
            target_layers = [modello.layer4]
            original_image = np.float32(image)
            input_tensor = preprocess_image(original_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            cam_algorithm = GradCAM
            targets = 65
            with cam_algorithm(model = modello, target_layers = target_layers) as cam:
                cam.batch_size = 32
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)

                # Here grayscale_cam has only one image in the batch
                grayscale_cam = grayscale_cam[0, :]
                mask = grayscale_cam * 255  #make range between 0-255
            
            plt.imshow(mask)
            plt.show()
            retval, img_thresh = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY) #threshold alla maschera per filtrare la zona focale
            mask = img_thresh.astype(np.uint8)

            img_applied_mask = cv2.bitwise_and(image, image, mask = mask)   #seziono l'immagine modificata con la maschera
            plt.imshow(img_applied_mask)
            plt.show()
            ###
            '''

            image = ifilter(image,*params[ilast:ilast+ifilter.nparams()])
            ilast += ifilter.nparams()
        return image

    def change(self, i, j, rand_params = False):
        p_i = 0
        for p in range(i):
            p_i += len(self.filters[self.genotype[p]].domains)
        e_i = p_i + len(self.filters[self.genotype[i]].domains)
        if rand_params == False:
            self.params = self.params[:p_i] + [d.value for d in self.filters[j].domains] + self.params[e_i:]
        else:
            self.params = self.params[:p_i] + [d.random() for d in self.filters[j].domains] + self.params[e_i:]
        self.genotype[i] = j 

    def pice(self, s=0, e=None):
        if e is None:
            e = len(self.genotype)
        new = copy.copy(self)
        new.fitness = new.fitness_max
        new.genotype = self.genotype[s:e]
        p_s = 0
        for i in range(s):
            p_s += len(self.filters[self.genotype[i]].domains)
        p_e = p_s
        for i in range(s,e):
            p_e += len(self.filters[self.genotype[i]].domains)
        new.params = self.params[p_s:p_e]
        return new

    def __add__(self, other):
        new = copy.copy(self)
        new.fitness = new.fitness_max
        new.genotype += other.genotype
        new.params += other.params
        return new
    
    def __len__(self):
        return len(self.genotype)

    def nparams(self):
        return len(self.params)