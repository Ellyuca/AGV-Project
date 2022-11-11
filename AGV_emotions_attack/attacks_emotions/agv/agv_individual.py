import copy
import random

from agv_gradcam_utils import *

class Individual(object):

    def __init__(self, 
                 Nf,
                 filters, 
                 fitness_max,
                 repetitions = True,
                 X = None):

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

        self.gradcam_mask_dict = self.create_mask_dict(X)
    

    def create_mask_dict(self, X):
        '''
        Create a dict as class attribute to save mask (and other img) which are used in filters application.
        These images are created for image X.
        This is made to make execution faster. Otherwhise these imgs are regenerated every times.
        '''
        mask, img_applied_mask, img_foreground, selected_pixels = self.gradcam_operations(X)
        gradcam_mask_dict = {'mask': mask, 'img_applied_mask': img_applied_mask, 'img_foreground': img_foreground, 'selected_pixels': selected_pixels}
        return gradcam_mask_dict
    

    def get_probabilistic_mask(self, grayscale_cam_mask, thresh_value, pct, thresh_method):
        '''
            Create a mask choosing probabilistically pixels from img_thresh with a percentage pct.
            
            params:
            - grayscale_cam_mask: grayscale image computed by grad_cam
            - thresh_value: value for BINARY threshold
            - pct: percentage of pixels to get (from the ones with value 255) for probabilistic mask
            
            returns:
            - new_mask: new img with selected pixels
        '''
        # Apply threshold
        _ , img_thresh = cv2.threshold(grayscale_cam_mask, thresh_value, 255, thresh_method) 
        mask = img_thresh.astype(np.uint8) #convert to uint8 for use in bitwise_and

        # Number of pixels to choose based on percentage value
        num_pixels = int(np.count_nonzero(mask > 0) / 100 * pct)

        # Define a new mask with zero value for all pixels
        new_mask = np.zeros(mask.shape)

        # Flatten images and define flatten indeces
        img_flat = mask.ravel()
        indices = np.arange(len(img_flat))

        # Normalize probability to sum 1 in order to choose only between pixels with value of 255
        normalized_prob = normalize(mask.ravel().reshape(1, -1), axis=1, norm='l1').ravel()

        # Random sample
        selected_pixels = np.column_stack(np.unravel_index(np.random.choice(indices, size=num_pixels, replace=False, p = normalized_prob), 
                                                            shape=mask.shape))
        # Apply in new mask the selected pixels
        for pixel in selected_pixels:
            new_mask[pixel[0], pixel[1]] = 255#need correction
        
        return new_mask, selected_pixels


    def gradcam_operations(self, image):
        '''
        Method used to make 3 images: mask, img_applied_mask, img_foreground.
        Start from an image, a grayscale_cam is calculated use GradCAM algorithm.
        From this grayscale_cam, is calculated (the method depends ):
            - mask -> area (pixels) where to apply filters (used as mask)
            - img_applied_mask -> area from the original img where to apply filters
            - img_foreground -> area where to not apply filters from the original img
        '''
        OG_class = None
        original_image = np.float32(image)
        input_tensor = preprocess_image(original_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        targets = OG_class
        cam_algorithm = GradCAM
        with cam_algorithm(model = MODEL_gradcam, target_layers = TARGET_LAYERS) as cam:
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]
            mask = grayscale_cam * 255  #make range between 0-255
        
        #_, img_thresh = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)
        Q_th_percentile = 80
        thresh_value = np.percentile(mask, Q_th_percentile)
        
        mask, selected_pixels = self.get_probabilistic_mask(mask, thresh_value=thresh_value, pct=50, thresh_method=cv2.THRESH_BINARY) #mask creation
        mask = mask.astype(np.uint8) #convert to uint8 for use in bitwise_and
        img_applied_mask = cv2.bitwise_and(image, image, mask = mask)
        img_logo_mask_inv = cv2.bitwise_not(mask)
        img_foreground = cv2.bitwise_and(image, image, mask = img_logo_mask_inv)

        return mask, img_applied_mask, img_foreground, selected_pixels


    def apply(self, image, params = None):
        if params is None:
            params = self.params
        ilast = 0
        #mask, img_applied_mask, img_foreground = self.gradcam_operations(image)
        mask = self.gradcam_mask_dict['mask']
        img_applied_mask = self.gradcam_mask_dict['img_applied_mask']
        img_foreground = self.gradcam_mask_dict['img_foreground']

        for fid in self.genotype:
            ifilter = self.filters[fid]
            # gradcam filters application
            image = ifilter(image,*params[ilast:ilast+ifilter.nparams()])
            img_applied_mask = cv2.bitwise_and(image, image, mask = mask)
            image = cv2.add(img_applied_mask, img_foreground)
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