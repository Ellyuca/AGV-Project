import sys, os
import PIL
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.preprocessing import image
VGG_MEAN = [103.939, 116.779, 123.68] # ?
from PIL import Image

from .resnet_model import ResNet_model
def keras_resnet_model():
    input_shape = (224, 224, 3)
    
    model = ResNet_model(input_shape=input_shape)
    return model
    
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    model = keras_emo_inception_resnet_v2_imagenet_model(x)