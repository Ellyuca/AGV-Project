import json
import pickle
import copy
import agv_individual
import base64

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
    
    def apply(self,X):
        ilast = 0 
        image = X
        for fid in self.model["filters"]:
            ifilter = self.model["filters_data"][fid]
            image = ifilter(image,*self.model["params"][ilast:ilast+ifilter.nparams()])
            ilast += ifilter.nparams()
        return image
    '''    
    def apply(self,X):
        model = models.resnet50(pretrained=True)
        target_layers = [model.layer4]

        origin_img = X * 255
        rgb_img = origin_img[:, :, ::-1]
        cv2.imwrite('rgb.JPEG', rgb_img)
        cv2.imwrite('bgr.JPEG', origin_img)
        
        origin_img = np.float32(origin_img) / 255
        input_tensor = preprocess_image(origin_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        cam_algorithm = GradCAM
        with cam_algorithm(model = model, target_layers = target_layers) as cam:
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]
            mask = grayscale_cam * 255  #make range between 0-255

        retval, img_thresh = cv2.threshold(mask, 160, 255, cv2.THRESH_BINARY)
        mask = img_thresh.astype(np.uint8)
        
        # Create colorful checkerboard background "behind" the logo lettering.
        img_applied_mask = cv2.bitwise_and(rgb_img, rgb_img, mask = mask)
        #cv2.imwrite('provola.JPEG', img_applied_mask)

        ilast = 0
        image = img_applied_mask # qui devo mettere la zona da me interessata
        #print('min',np.min(image))
        #print(np.max(X))
        #image = X
        for fid in self.model["filters"]:
            ifilter = self.model["filters_data"][fid]
            image = ifilter(image,*self.model["params"][ilast:ilast+ifilter.nparams()])
            ilast += ifilter.nparams()
        

        image = cv2.bitwise_and(image, image, mask= mask)
        #print('min ', np.min(image))

        cv2.imwrite('filtro.JPEG', image*255)
        # qui devo rifare l'applicazione della maschera
        img_logo_mask_inv = cv2.bitwise_not(mask)
        img_foreground = cv2.bitwise_and(rgb_img, rgb_img, mask = img_logo_mask_inv)
        #cv2.imwrite('provola.JPEG', image*255)
        result = cv2.add(image,img_foreground/255)
        #cv2.imwrite('provola.JPEG', result)
        #print(np.max(result))
        return result[:, :, ::-1]
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