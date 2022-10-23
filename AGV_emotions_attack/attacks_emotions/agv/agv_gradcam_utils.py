import cv2
import numpy as np
from torchvision import models
from pytorch_grad_cam import GradCAM  
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt

MODEL_gradcam = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
TARGET_LAYERS = [MODEL_gradcam.layer4]