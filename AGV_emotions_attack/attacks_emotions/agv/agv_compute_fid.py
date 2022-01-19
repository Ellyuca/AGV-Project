import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from scipy.linalg import sqrtm


# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		new_image = resize(image, new_shape, 0)
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

class ComputeFid:
    def __init__(self):
        self.model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

    def __call__(self, images1, images2):
        # convert integer to floating point values
        images1 = images1.astype('float32')
        images2 = images2.astype('float32')
        # resize images
        if images1.shape[1] != 299 or images1.shape[2] != 299 or images1.shape[3] != 3:
            images1 = scale_images(images1, (299,299,3))
        if images2.shape[1] != 299 or images2.shape[2] != 299 or images2.shape[3] != 3:
            images2 = scale_images(images2, (299,299,3))
        return calculate_fid(self.model,images1,images2)
