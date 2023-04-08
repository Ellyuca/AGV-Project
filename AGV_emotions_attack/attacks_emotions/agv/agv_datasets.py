import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(os.path.join(os.path.dirname(__file__),'..', ".."))


from datasets.images import Imagenet

DATASET_ID = 0
MODEL_ID = 1

def build_model():
    info_db_fq = database_and_model()["IMAGENET-RESNET"]
    dataset = info_db_fq[DATASET_ID]
    model = dataset.load_model_by_name(info_db_fq[MODEL_ID])
    return model

def build_model_and_dataset(dataset_name):
    info_db_fq = database_and_model()[dataset_name]
    dataset = info_db_fq[DATASET_ID]   

    X, Y = dataset.get_test_dataset()
    model = dataset.load_model_by_name(info_db_fq[MODEL_ID])
    
    return model, X, Y


class DatasetSubset:
    def __init__(self, BaseClass, nsamples = 'all', start_nsamples=0):
        self.base_class = BaseClass()
        self.name = self.base_class.__class__.__name__
        self.start_nsamples = start_nsamples
        self.nsamples = nsamples
        if nsamples != 'all':
            self.stop_nsamples = start_nsamples + nsamples

    def get_test_dataset(self):
        if self.name == "Imagenet":
            affectnet_max_images = 50000
            if self.nsamples == 'all' and self.start_nsamples > 0:
                X_train, Y_train = self.base_class.get_test_dataset(num_images=affectnet_max_images)
                return X_train[self.start_nsamples,:,:,:],Y_train[self.start_nsamples,:,:,:]
            elif self.nsamples == 'all':
                return self.base_class.get_test_dataset(num_images=affectnet_max_images)
            elif self.start_nsamples > 0:
                X_train, Y_train = self.base_class.get_test_dataset(num_images=affectnet_max_images)
                print(X_train.shape, Y_train.shape) #(200, 224, 224, 3) (200, 1000)
                print(X_train[self.start_nsamples:self.stop_nsamples].shape, Y_train[self.start_nsamples:self.stop_nsamples].shape)
                #return X_train[self.start_nsamples:self.stop_nsamples,:,:,:], Y_train[self.start_nsamples:self.stop_nsamples,:,:,:]
                return X_train[self.start_nsamples:self.stop_nsamples], Y_train[self.start_nsamples:self.stop_nsamples]
            else:
                return self.base_class.get_test_dataset(num_images=self.nsamples)
        else:
            if self.nsamples == 'all':
                X_train, Y_train = self.base_class.get_test_dataset()
                return X_train[self.start_nsamples:], Y_train[self.start_nsamples:]
            else:
                X_train, Y_train = self.base_class.get_test_dataset()
                return X_train[self.start_nsamples:self.stop_nsamples], Y_train[self.start_nsamples:self.stop_nsamples]



    def get_val_dataset(self):
        return self.base_class.get_val_dataset()

    def load_model_by_name(self, *args,**kargs):
        return self.base_class.load_model_by_name(*args,**kargs)

def database_and_model():
    return {

        'IMAGENET-RESNET' : [
            DatasetSubset(Imagenet, nsamples = 200, start_nsamples=0),
            'resnet',       
        ],            

    }    

def get_model_name_from_dataset(dataset_name):
    try:
        return database_and_model()[dataset_name][MODEL_ID]
    except:
        return 'unkown'
