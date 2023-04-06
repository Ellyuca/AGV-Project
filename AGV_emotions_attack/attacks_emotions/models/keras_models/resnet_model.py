import sys, os
# from turtle import shape
from xml.etree.ElementInclude import include
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_externals
import pathlib
import tensorflow as tf

'''
#classic definition of a model, if use this do preprocessing of input
def MobileNet_model(input_shape):
  base_model = tf.keras.applications.MobileNet()
  return base_model
'''
'''
def MobileNet_model(input_shape):
  inputs = tf.keras.Input(shape=input_shape)
  x = tf.keras.applications.mobilenet.preprocess_input(inputs*255)
  
  core = tf.keras.applications.MobileNet()
  x = core(x)
  
  model = tf.keras.Model(inputs=[inputs], outputs=[x])

  return model
'''

def ResNet_model(input_shape, download_model = False):
  inputs = tf.keras.Input(shape=input_shape)
  x = tf.keras.applications.resnet50.preprocess_input(inputs*255)

  core = tf.keras.applications.ResNet50(weights=None)
  x = core(x)

  model = tf.keras.Model(inputs=[inputs], outputs=[x], name='resnet50')
  
  if download_model:
    model.save_weights(os.path.join(pathlib.Path(__file__).parent.absolute(), "../resnet50_weights/keras_resnet50_weights.h5"))
  else:
    model.load_weights(os.path.join(pathlib.Path(__file__).parent.absolute(), "../resnet50_weights/keras_resnet50_weights.h5"))

  return model