import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import uproot
import utils
import yaml
import torch

class Data_Fetcher():
    def __init__(self):
        self.data_train = None
        self.data_test = None
        
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

        def keras_to_torch(X, y):
            X_torch = torch.tensor(X.transpose(0, 3, 1, 2), dtype=torch.float32)
            y_torch = torch.tensor(y, dtype=torch.long)
            return X_torch, y_torch

        feature_array, y, spec_array = utils.get_features_labels('root://eospublic.cern.ch//eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/train/ntuple_merged_10.root', features, spectators, labels, remove_mass_pt_window=False, entry_stop=10000)
        # make image
        X = utils.make_image(feature_array)
        # image is a 4D tensor (n_samples, n_pixels_x, n_pixels_y, n_channels)

        X_train, y_train = keras_to_torch(X, y)

        # load testing file
        feature_array_test, label_array_test, spec_array_test = utils.get_features_labels('root://eospublic.cern.ch//eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/test/ntuple_merged_0.root', features, spectators, labels, remove_mass_pt_window=False)
                      
        # make image
        X_test = utils.make_image(feature_array_test)
        X_test, y_test = keras_to_torch(X_test, label_array_test)
    
        self.data_train = torch.utils.data.TensorDataset(torch.stack(X_train), torch.stack(y_train))
        self.data_test = torch.utils.data.TensorDataset(torch.stack(X_test), torch.stack(y_test))
        
        return X_train, y_train, X_test, y_test

    
if __name__ == '__main__':
    
    fetcher = Data_Fetcher()
    x,y, _, _ = fetcher.get_data()
    
    print(x)