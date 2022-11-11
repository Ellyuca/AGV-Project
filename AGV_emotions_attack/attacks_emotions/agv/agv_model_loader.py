import json
import pickle
import copy
import agv_individual
import base64

from agv_gradcam_utils import *

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
        
        sub_path = path.replace('best_jsons', 'gradcam_mask_dict')
        self.gradcam_mask_dict['selected_pixels'] = self.load_dict(sub_path)
        return self
    
    def load_dict(self,path): #funzione per caricare il dizionare dal json con i selected pixel
        with open(path, 'r') as jfile:
            jdict = json.load(jfile)
            self.gradcam_mask_dict = copy.copy(jdict)
            self.gradcam_mask_dict["selected_pixels"] = np.asarray(jdict["selected_pixels"])

        return self.gradcam_mask_dict['selected_pixels']

    def save(self,path):
        with open(path, 'w') as jfile:
            jmodel = copy.copy(self.model)
            filters_data_b64 = base64.b64encode(pickle.dumps(self.model["filters_data"]))
            jmodel["filters_data"] = filters_data_b64.decode("ascii")
            json.dump(jmodel, jfile, indent=4)
        return self
    
    def save_dict(self,path): #funzione per salvare il dizionario in json con i selected pixel
        with open(path, 'w') as jfile:
            jdict = copy.copy(self.gradcam_mask_dict['selected_pixels'].tolist())
            selected_pixels_dict = {'selected_pixels': jdict}
            json.dump(selected_pixels_dict, jfile, indent=4)
        return self 

    def gradcam_operations(self, image):
        '''
        Method used to make 3 images: mask, img_applied_mask, img_foreground.
        Start from an image, a grayscale_cam is calculated use GradCAM algorithm.
        From this grayscale_cam, is calculated (the method depends ):
        - mask -> area (pixels) where to apply filters (used as mask)
        - img_applied_mask -> area from the original img where to apply filters
        - img_foreground -> area where to not apply filters from the original img
        '''
        # Define a new mask with zero value for all pixels
        new_mask = np.zeros((image.shape[0], image.shape[1]))
        # Apply in new mask the selected pixels
        for pixel in self.gradcam_mask_dict['selected_pixels']:
            new_mask[pixel[0], pixel[1]] = 255#need correction

        new_mask = new_mask.astype(np.uint8) #convert to uint8 for use in bitwise_and
        img_applied_mask = cv2.bitwise_and(image, image, mask = new_mask)
        img_logo_mask_inv = cv2.bitwise_not(new_mask)
        img_foreground = cv2.bitwise_and(image, image, mask = img_logo_mask_inv)

        return new_mask, img_applied_mask, img_foreground

    def apply(self, image):
        ilast = 0
        mask, img_applied_mask, img_foreground = self.gradcam_operations(image)        
        for fid in self.model["filters"]:
            ifilter = self.model["filters_data"][fid]
            #filters application
            image = ifilter(image,*self.model["params"][ilast:ilast+ifilter.nparams()])
            img_applied_mask = cv2.bitwise_and(image, image, mask = mask)
            image = cv2.add(img_applied_mask, img_foreground)
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
    
    def from_individual_dict(self, individual):
        #funzione per ottenere il dizionario e salvarlo nel giusto attributo in self
        self.gradcam_mask_dict = individual.gradcam_mask_dict
        return self