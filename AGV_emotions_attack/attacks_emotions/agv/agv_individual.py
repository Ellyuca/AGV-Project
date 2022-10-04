import copy
import random

###
import cv2
import numpy as np
from torchvision import models
#from agv_attack import MODEL_gradcam, TARGET_LAYERS
MODEL_gradcam = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
TARGET_LAYERS = [MODEL_gradcam.layer4]
from pytorch_grad_cam import GradCAM  
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import preprocess_image
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
    

    def gradcam_operations(self, image):
        original_image = np.float32(image)
        input_tensor = preprocess_image(original_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        targets = [ClassifierOutputTarget(67)]
        cam_algorithm = GradCAM
        with cam_algorithm(model = MODEL_gradcam, target_layers = TARGET_LAYERS) as cam:
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]
            mask = grayscale_cam * 255  #make range between 0-255

        
        _, img_thresh = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)
        mask = img_thresh.astype(np.uint8) #convert to uint8 for use in bitwise_and
        img_applied_mask = cv2.bitwise_and(image, image, mask = mask)
        img_logo_mask_inv = cv2.bitwise_not(mask)
        img_foreground = cv2.bitwise_and(image, image, mask = img_logo_mask_inv)

        #plt.imshow(img_applied_mask, cmap='gray')
        #plt.show()

        return mask, img_applied_mask, img_foreground


    def apply_gradcam(self, image, params = None):
        if params is None:
            params = self.params
        ilast = 0

        mask, img_applied_mask, img_foreground = self.gradcam_operations(image)
        
        for fid in self.genotype:
            ifilter = self.filters[fid]
            image = ifilter(image,*params[ilast:ilast+ifilter.nparams()])
            #plt.imshow(image)
            #plt.show()

            img_applied_mask = cv2.bitwise_and(image, image, mask = mask)
            #plt.imshow(img_applied_mask)
            #plt.show()

            image = cv2.add(img_applied_mask,img_foreground)
            ilast += ifilter.nparams()

        #plt.imshow(image)
        #plt.show()
        return image


    def apply(self, image, params = None):
        if params is None:
            params = self.params
        ilast = 0
        for fid in self.genotype:
            ifilter = self.filters[fid]
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