import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import errno
import pathlib
import random
import json
import numpy as np
from PIL import Image
from PIL import ImageDraw
from filters import show_image
from filters import _to_pil_image
from agv_model_loader import ModelLoader
from agv_datasets import build_model_and_dataset
from agv_metrics import compute_metricts
import pandas as pd
import tensorflow as tf

from agv_xai_utils import *


# load classes json data
json_file = open(os.path.join(os.path.join(pathlib.Path(__file__).parent.absolute(), 
									  "../datasets/imagenet_class_index.json")))
class_names = json.load(json_file)

def get_cam(img):
	img = np.float32(img)
	input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	cam_algorithm = EigenCAM
	with cam_algorithm(model = MODEL_gradcam, target_layers = TARGET_LAYERS, use_cuda = True) as cam:
		cam.batch_size = 256
		grayscale_cam_eigen_original = cam(input_tensor=input_tensor, targets=None)
		image_cam = grayscale_cam_eigen_original[0, :]
	return image_cam

def get_center(coordinates):
	cx = int((coordinates[0] + coordinates[2]) / 2)
	cy = int((coordinates[1] + coordinates[3]) / 2)
	return (cx, cy)

def get_cam_on_image(rgb_img, cam):
	cam_on_image = show_cam_on_image(rgb_img, cam, use_rgb=True)
	return cam_on_image

def one_sample_batch(image):
	batch = np.zeros(shape=(1,*image.shape))
	batch[0,:,:,:]=image
	return batch

def predict_image(model,image):
	return np.argmax(model.predict(one_sample_batch(image))[0])

def predict_image_detailed(model,image):
	return model.predict(one_sample_batch(image))[0]

def dataset_from_model(path_model):
	import json 
	with open(path_model,"r") as jfile:
		return json.load(jfile)["in_params"]["dataset"]

def test_fits(path_model, dataset_name = None):
	if dataset_name is None:
		dataset_name = dataset_from_model(path_model)
	nn_model, X, Y = build_model_and_dataset(dataset_name)
	agv_model = ModelLoader().load(path_model).to_individual()
	#compute metricts
	metricts = compute_metricts(agv_model, nn_model, dataset_name, X, Y)
	#print
	print("Best fit:", agv_model.fitness)
	print("Best genotype:", agv_model.genotype)
	print("Best params:", agv_model.params)
	for key,val in metricts.items():
		print("{}:".format(key), val)

def test(path_model, show_info = False, dataset_name = None):
	if dataset_name is None:
		dataset_name = dataset_from_model(path_model)
	nn_model, X, Y = build_model_and_dataset(dataset_name)
	I = random.randrange(0,X.shape[0])
	model = ModelLoader().load(path_model)
	blank_image = Image.new("RGB",(X.shape[1]*2+3,X.shape[2]+2))
	blank_image.paste(_to_pil_image(X[I]),(1,1)) 
	blank_image.paste(_to_pil_image(model.apply(X[I])),(X.shape[1]+2,1)) 
	print("predict p1 in test function:")
	p1 = predict_image(nn_model, X[I])
	print("predict p2 in test function:")
	p2 = predict_image(nn_model, model.apply(X[I]))
	draw = ImageDraw.Draw(blank_image)
	draw.text((0, 0), "{}".format(p1), (255, 0, 0))
	draw.text((X.shape[1]+1, 0), "{}".format(p2), (255, 0, 0))
	blank_image.show()
	if show_info:
		gf = model.get_filters()
		filters = str([str(fp[0].name)+str(fp[1:]) for fp in gf])
		print(str(filters))
	
def mkdir_p(dirname):
	try:
		os.mkdir(dirname)
	except OSError as exc:
		if exc.errno != errno.EEXIST:
			raise
		pass

def save_adv_ex(path_model, nimages = 10, dataset_name = None):
	if dataset_name is None:
		dataset_name = dataset_from_model(path_model)
	nn_model, X, Y = build_model_and_dataset(dataset_name)
	print(dataset_name, "size:", X.shape[0])
	model = ModelLoader().load(path_model)
	P = os.path.splitext(path_model)[0]
	mkdir_p(P)
	P_t = os.path.join(P,"img{}_{}_{}.png")
	N = 0
	for I in range(X.shape[0]):
		
		p1 = predict_image(nn_model, X[I])
		print("predict p1 in save_adv_ex function:", p1)
		
		p2 = predict_image(nn_model, model.apply(X[I]))
		print("predict p2 in save_adv_ex function:", p2)
		if p1 != p2:
			blank_image = Image.new("RGB",(X.shape[1]*2+3,X.shape[2]+2))
			blank_image.paste(_to_pil_image(X[I]),(1,1)) 
			blank_image.paste(_to_pil_image(model.apply(X[I])),(X.shape[1]+2,1)) 
			draw = ImageDraw.Draw(blank_image)
			draw.text((0, 0), "{}".format(p1), (255, 0, 0))
			draw.text((X.shape[1]+1, 0), "{}".format(p2), (255, 0, 0))
			blank_image.save(P_t.format(I,p1,p2))
			N += 1
			if nimages <= N: 
				return 0
  
def save_adv_best(best_folder, image_id=0, dataset_name = None ):

	best_files = os.listdir(best_folder)
	best_files.sort(key=lambda s: int(s.split("_")[1]))
	#TO DO : condition when there isnt a best json for that image id
	print("best_files:", best_files)
	if dataset_name is None:
		dataset_name = dataset_from_model(os.path.join(best_folder, best_files[image_id]))

	nn_model, X, Y = build_model_and_dataset(dataset_name)
	best_model =  ModelLoader().load(os.path.join(best_folder, best_files[image_id]))
	print("Starting saving best" )
	print(dataset_name, "size:", X.shape[0], "network name: ", nn_model.name)

	P = (os.path.splitext(best_folder)[0]).split("/")[-2]  
	mkdir_p(P)
	P = os.path.join(P, "generated_best_images")
	mkdir_p(P) 
	P_t = os.path.join(P,"best_img{}_.png")
	blank_image = Image.new("RGB",(224,224))
	blank_image.paste(_to_pil_image(best_model.apply(X[image_id])))
	#cv2.imwrite('TEST'+str(image_id)+'.png',best_model.apply(X[image_id])) #used to check selected pixels
	draw = ImageDraw.Draw(blank_image)                
	blank_image.save(P_t.format(image_id))

	p1 = predict_image(nn_model, X[image_id])
	p2 = predict_image(nn_model, best_model.apply(X[image_id]))

	P_tt = os.path.join(P,"OG_vs_best_img{}_{}_{}.png")
	blank_image = Image.new("RGB",(X.shape[1]*2+3,X.shape[2]+2))
	blank_image.paste(_to_pil_image(X[image_id]),(1,1))
	blank_image.paste(_to_pil_image(best_model.apply(X[image_id])),(X.shape[1]+2,1)) 
	draw = ImageDraw.Draw(blank_image)
	draw.text((0, 0), "{}".format(p1), (255, 0, 0))
	draw.text((X.shape[1]+1, 0), "{}".format(p2), (255, 0, 0))
	blank_image.save(P_tt.format(image_id,p1,p2))

	#create img for cam
	P_tt = os.path.join(P,"best_img_cam{}_.png")
	blank_image = Image.new("RGB",(X.shape[1]*2+3,X.shape[2]+2))
	cam_image_original = np.expand_dims(get_cam(X[image_id]), 2)
	cam_image_modified = np.expand_dims(get_cam(best_model.apply(X[image_id])), 2)
	blank_image.paste(_to_pil_image(cam_image_original),(1,1)) 
	blank_image.paste(_to_pil_image(cam_image_modified),(X.shape[1]+2,1))
	draw = ImageDraw.Draw(blank_image)
	draw.text((0, 0), "{}".format('original'), (255, 0, 0))
	draw.text((X.shape[1]+1, 0), "{}".format('attacked'), (255, 0, 0))
	blank_image.save(P_tt.format(image_id))

	P_tt = os.path.join(P,"best_img_cam_on_image{}_.png")
	blank_image = Image.new("RGB",(X.shape[1]*2+3,X.shape[2]+2))
	cam_on_image_original = np.float32(get_cam_on_image(X[image_id], cam_image_original)) / 255
	cam_on_image_modified = np.float32(get_cam_on_image(X[image_id], cam_image_modified)) / 255
	blank_image.paste(_to_pil_image(cam_on_image_original),(1,1)) 
	blank_image.paste(_to_pil_image(cam_on_image_modified),(X.shape[1]+2,1))
	draw = ImageDraw.Draw(blank_image)
	draw.text((0, 0), "{}".format('original'), (255, 0, 0))
	draw.text((X.shape[1]+1, 0), "{}".format('attacked'), (255, 0, 0))
	blank_image.save(P_tt.format(image_id))

	#create IoU comparison
	P_tt = os.path.join(P,"img_IoU_comparison_{}_.png")
	blank_image = Image.new("RGB",(224,224))
	cam_image_original = np.uint8(cam_image_original * 255)
	cam_image_modified = np.uint8(cam_image_modified * 255)
	_, thresh_original_image = cv2.threshold(cam_image_original, 170, 255, cv2.THRESH_BINARY)
	_, thresh_modified_image = cv2.threshold(cam_image_modified, 170, 255, cv2.THRESH_BINARY)
	contours_original_image, _ = cv2.findContours(thresh_original_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours_modified_image, _ = cv2.findContours(thresh_modified_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	x_o,y_o,w_o,h_o = cv2.boundingRect(contours_original_image[0])
	x_m,y_m,w_m,h_m = cv2.boundingRect(contours_modified_image[0])
	cv2.rectangle(cam_on_image_modified, (x_o, y_o), (x_o + w_o, y_o + h_o), (0,1,0), 2)
	cv2.rectangle(cam_on_image_modified, (x_m, y_m), (x_m + w_m, y_m + h_m), (1,0,0), 2)
	#draw center point of IoU
	center_original = get_center((x_o, y_o, x_o + w_o, y_o + h_o))
	center_modified = get_center((x_m, y_m, x_m + w_m, y_m + h_m))
	cv2.circle(cam_on_image_modified, center_original, radius=1, color=(0, 1, 0), thickness=-1)
	cv2.circle(cam_on_image_modified, center_modified, radius=1, color=(1, 0, 0), thickness=-1)
	blank_image.paste(_to_pil_image(cam_on_image_modified))
	draw = ImageDraw.Draw(blank_image)
	draw.text((0, 0), "{}".format('original'), (0, 255, 0))
	draw.text((0, 10), "{}".format('modified'), (255, 0, 0))
	blank_image.save(P_tt.format(image_id))