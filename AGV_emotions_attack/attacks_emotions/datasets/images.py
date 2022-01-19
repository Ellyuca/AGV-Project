import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pathlib
import numpy as np
import os

from tensorflow.keras.preprocessing import image
from models.keras_models import keras_emo_mobilenetv2_model
# pool = Pool()

def load_single_image(img_path, img_size=224):
    size = (img_size,img_size)
    img = image.load_img(img_path, target_size=size, interpolation="nearest") 
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)
    # Embeded preprocessing in the model.
    # x = preprocess_input(x)
    return x


def _load_single_image(args):
    img_path, img_size = args
    return load_single_image(img_path, img_size)


def data_images(img_folder, img_size, label_style = 'caffe', label_size = 8, selected_idx = None):
    fnames = os.listdir(img_folder)
    fnames = sorted(fnames, key = lambda x: int(x.split('.')[1]))
    print("images names: ", fnames)
    
    if isinstance(selected_idx, list):
        selected_fnames = [fnames[i] for i in selected_idx]
    elif isinstance(selected_idx, int):
        selected_fnames = fnames[:selected_idx]
    else:
        selected_fnames = fnames

    labels = list(map(lambda x: int(x.split('.')[0]), selected_fnames))
    img_path_list = map(lambda x: [os.path.join(img_folder, x), img_size], selected_fnames)
    X = list(map(_load_single_image, img_path_list))
    X = np.concatenate(X, axis=0)
    Y = np.eye(8)[labels]
    return X, Y


class AffectNetDataset:
    def __init__(self):
        self.dataset_name = "AffectNet"
        # self.image_size = 224
        self.num_channels = 3
        self.img_folder = os.path.join(pathlib.Path(__file__).parent.absolute(),
                          "../datasets/images_dataset/emo_images/selected_balanced_images_renamed")
        if not os.path.isdir:
            raise Exception("Please prepare the dataset first")

    def get_test_dataset(self, img_size=224, num_images=100):
        self.image_size = img_size
        X, Y = data_images(self.img_folder, self.image_size, selected_idx=num_images)
        X /= 255
        return X, Y



    def load_model_by_name(self, model_name):
 
        if model_name == 'emo_mobilenet_v2':
            model = keras_emo_mobilenetv2_model() 
        else:
            raise Exception("Unsupported model: [%s]" % model_name)

        return model

if __name__ == '__main__':
    dataset = AffectNetDataset()

    X, Y = dataset.get_test_dataset()
    model = dataset.load_model_by_name('emo_mobilenet_v2')


    

