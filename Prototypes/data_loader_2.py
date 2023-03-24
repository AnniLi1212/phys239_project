import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import uproot
import utils
import yaml
import torch
import torch.nn.functional as F


class Data_Fetcher():
    def __init__(self):
        self.data_train = None
        self.data_test = None
        self.input_shape = None
        
    def get_data(self):

        with open('definitions_image.yml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
            definitions = yaml.load(file, Loader=yaml.FullLoader)
    
        features = definitions['features']
        spectators = definitions['spectators']
        labels = definitions['labels']

        nfeatures = definitions['nfeatures']
        nspectators = definitions['nspectators']
        nlabels = definitions['nlabels']
        
        def keras_to_torch(X, y):  #this may not be correct, X and y are already 2D somehow
            X_torch = torch.tensor(X.transpose(0, 3, 1, 2), dtype=torch.float32)
            y_torch = torch.tensor(y, dtype=torch.long)
            return X_torch, y_torch

        feature_array, y, spec_array = utils.get_features_labels('root://eospublic.cern.ch//eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/train/ntuple_merged_10.root', features, spectators, labels, remove_mass_pt_window=False, entry_stop=1000)
        print(1)
        
        
        
        # make image
        X = utils.make_image(feature_array) 
        
        # image is a 4D tensor (n_samples, n_pixels_x, n_pixels_y, n_channels)
        print(2)
        print(y)
        X_train, y_train = keras_to_torch(X, y[:,0])  # picking higgs events, dim 0
        
        
        y_train = y_train.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        y_train = y_train.repeat(1,1,224,224)

       
        print(3)
        self.input_shape = X_train.shape   #adjust if data changes
        print(X_train.shape)
        print(y_train.shape)
   
        print(4)
        X_test = X_train
        y_test = y_train
        print(5)

        self.data_train = torch.utils.data.TensorDataset(torch.cat((X_train.float(), y_train), dim=1))
        print(6)
        self.data_test = torch.utils.data.TensorDataset(X_test)#torch.stack(X_test), torch.stack(y_test))
        print(self.data_train)
        return X_train, y_train, X_test, y_test

    
if __name__ == '__main__':
    
    fetcher = Data_Fetcher()
    x,y, _, _ = fetcher.get_data()
    
    print(x)