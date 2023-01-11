import cv2
import numpy as np
from torchvision.models import resnet50
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image

from sklearn.preprocessing import normalize

MODEL_gradcam = resnet50(pretrained=True)
TARGET_LAYERS = [MODEL_gradcam.layer4]