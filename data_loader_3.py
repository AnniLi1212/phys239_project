import os
import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import uproot
import utils
import yaml
import torch
import h5py
import tables
import keras
import awkward as ak


class Data_Fetcher():
    def __init__(self):
        self.data_train = None
        self.data_test = None


    def get_features_labels(self,file_name, remove_mass_pt_window=True):

        with open('definitions_image.yml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
            definitions = yaml.load(file, Loader=yaml.FullLoader)

        #The features being obeserved in the CNN    
        features = definitions['features']

        spectators = definitions['spectators']
        labels = ['fj_isQCD*sample_isQCD',
          'fj_isH*fj_isBB']

        nfeatures = 3
        nspectators = 2
        nlabels = 2

        # load file
        h5file = tables.open_file(file_name, 'r')
        njets = getattr(h5file.root,features[0]).shape[0]

        # allocate arrays
        feature_array = np.zeros((njets,nfeatures))
        spec_array = np.zeros((njets,nspectators))
        label_array = np.zeros((njets,nlabels))

        # load feature arrays
        for (i, feat) in enumerate(features):
            ak_array = ak.from_iter(getattr(h5file.root, feat)[:])
            feature_array[:, i] = utils.to_np_array(ak_array, max_n=100, pad=0)

        # load spectator arrays
        for (i, spec) in enumerate(spectators):
            spec_array[:,i] = getattr(h5file.root,spec)[:]

        # load labels arrays
        for (i, label) in enumerate(labels):
            prods = label.split('*')
            prod0 = prods[0]
            prod1 = prods[1]
            fact0 = getattr(h5file.root,prod0)[:]
            fact1 = getattr(h5file.root,prod1)[:]
            label_array[:,i] = np.multiply(fact0,fact1)

        return feature_array, label_array, spec_array



    def keras_to_torch(self, X, y):
        X_torch = torch.tensor(X.transpose(0, 3, 1, 2), dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.long)
        return X_torch, y_torch


    def get_data(self):
        if not os.path.isfile('ntuple_merged_10.h5'):
            os.system('root://eospublic.cern.ch//eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/train/ntuple_merged_10.root')

        feature_array, y, spec_array = self.get_features_labels('ntuple_merged_10.h5', remove_mass_pt_window=False)

        # make image
        X = np.stack([utils.make_image(feature_array) for feature_array in feature_arrays], axis=-1)
        # image is a 4D tensor (n_samples, n_pixels_x, n_pixels_y, n_channels)
        X_train, y_train = self.keras_to_torch(X, y)


        if not os.path.isfile('ntuple_merged_0.h5'):
            os.system('root://eospublic.cern.ch//eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/test/ntuple_merged_0.root.')

        feature_array_test, label_array_test, spec_array_test = self.get_features_labels('ntuple_merged_0.h5', remove_mass_pt_window=False)


        # make image
        X_test = utils.make_image(feature_array_test)
        X_test, y_test = self.keras_to_torch(X_test, label_array_test)


        self.data_train = torch.utils.data.TensorDataset(torch.stack(X_train), torch.stack(y_train))
        self.data_test = torch.utils.data.TensorDataset(torch.stack(X_test), torch.stack(y_test))


        return X_train, y_train, X_test, y_test

    
if __name__ == '__main__':
    
    fetcher = Data_Fetcher()
    x, y, _, _ = fetcher.get_data()
    
    print(x)