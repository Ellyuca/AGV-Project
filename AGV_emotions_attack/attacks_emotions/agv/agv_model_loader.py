import json
import pickle
import copy
import agv_individual
import base64

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

class ModelLoader(object):

    def __init__(self):
        self.model = None 

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
    

    def gradcam_operations(self, image, OG_class=None):
        original_image = np.float32(image)
        input_tensor = preprocess_image(original_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        targets = [ClassifierOutputTarget(OG_class)] if OG_class != None else None
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


    def apply(self, image, OG_class=None):
        ilast = 0 
        
        mask, img_applied_mask, img_foreground = self.gradcam_operations(image, OG_class)
                
        for fid in self.model["filters"]:
            ifilter = self.model["filters_data"][fid]
            image = ifilter(image,*self.model["params"][ilast:ilast+ifilter.nparams()])
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

    '''def apply(self,X):#???????????????
        ilast = 0 
        image = X
        for fid in self.model["filters"]:
            ifilter = self.model["filters_data"][fid]
            image = ifilter(image,*self.model["params"][ilast:ilast+ifilter.nparams()])
            ilast += ifilter.nparams()
        return image
    '''

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