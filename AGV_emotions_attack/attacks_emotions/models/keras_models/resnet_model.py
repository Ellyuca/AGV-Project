import sys, os
from turtle import shape
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

def ResNet_model(input_shape):
  inputs = tf.keras.Input(shape=input_shape)
  x = tf.keras.applications.resnet50.preprocess_input(inputs*255)

  core = tf.keras.applications.ResNet50()
  x = core(x)

  model = tf.keras.Model(inputs=[inputs], outputs=[x])

  return model

'''
def EMO_MobileNetV2_model(input_shape): #OLD MODEL DEFINITION

  base_model = MobileNetV2(weights=None, include_top=False, pooling='avg')

  inputs = tensorflow.keras.Input(shape=input_shape)
  x = preprocess_mobilenetv2(inputs*255)
  network = base_model(x)
  network = tensorflow.keras.layers.Dense(128, activation="relu")(network)
  network = tensorflow.keras.layers.Dropout(.5)(network)
  outputs = tensorflow.keras.layers.Dense(8, activation = tensorflow.keras.activations.softmax)(network)

  EMO_MobileNetV2_model = tensorflow.keras.Model(inputs, outputs, name='emo_mobilenet_v2')
  EMO_MobileNetV2_model.load_weights(os.path.join(pathlib.Path(__file__).parent.absolute(), 
                          "../emo_weights/Emotion_MobileNetV2_weights.h5"))

  return EMO_MobileNetV2_model
'''