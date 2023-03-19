#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:07:23 2023

@author: daniel
"""
import caffe
from caffe import layers
import numpy as np

class DataReader:
    def __init__(self, root_file, batch_size):
        self.root_file = root_file
        self.batch_size = batch_size

    def read_data(self):
        # TODO: Implement reading of data from root file
        pass

class CVNModel:
    def __init__(self, input_shape, num_classes):
        self.net = caffe.NetSpec()
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create_network(self):
        self.net.data = layers.Input(name='data path', shape=[2, 3, 224, 224]) # do we use shape= self.input_shape ?

        self.net.conv1 = layers.Convolution(self.net.data, name='conv1', kernel_size=7, stride=2)
        self.net.pool1 = layers.Pooling(self.net.conv1, name='pool1', kernel_size=3, stride=2)  
        self.net.lrn1 = layers.LRN(self.net.pool1, name='lrn1') #figure out required inputs here
        self.net.conv2 = layers.Convolution(self.net.lrn1,name='conv2', kernel_size=1)
        self.net.conv3 = layers.Convolution(self.net.conv2, name='conv3', kernel_size=3)
        self.net.lrn2 = layers.LRN(self.net.conv3, name='lrn2')
        self.net.pool2 = layers.Pooling(self.net.lrn2, name='pool2', kernel_size=3, stride=2)
        
        #inception modules
        self.net.inc1 = Inception_Module(self.net.pool2, 'inc1')
        self.net.inc2 = Inception_Module(self.net.inc1, 'inc2')
        
        self.net.pool3 = layers.Pooling(self.net.inc2, name='pool3', kernel_size = 3, stride=2)
        
        # more inception modules
        self.net.inc3 = Inception_Module(self.net.pool3, 'inc3')
        
        self.net.pool4 = layers.Pooling(self.net.inc3, name='pool4', kernel_h=6, kernel_w=5, pool='AVE')
        self.net.softmax = layers.softmax(self.net.pool4, name='softmax')

class Inception_Module:
    def __init__(self, prev_layer, name): # this might need to be fleshed out better
        self.net = caffe.NetSpec()
        self.prev_layer = prev_layer
        
        self.net.conv_a1 = layers.Convolution(self.prev_layer, name='conv_a1',kernel_size=1)
        self.net.conv_b1 = layers.Convolution(self.prev_layer, name='conv_b1',kernel_size=1)
        self.net.conv_c1 = layers.Convolution(self.prev_layer, name='conv_c1',kernel_size=1)
        self.net.pool_d1 = layers.Pooling(self.prev_layer, name='pool_d1', kernel_size=3)
        
        self.net.conv_b2 = layers.Convolution(self.net.conv_b1, name='conv_b2',kernel_size=3)
        self.net.conv_c2 = layers.Convolution(self.net.conv_c1, name='conv_c2',kernel_size=5)
        self.net.conv_d2 = layers.Convolution(self.net.pool_d1, name='conv_d2',kernel_size=1)


        # TODO: how to implement filter concatenation? This will pull all branches together and give an output

        self.net.output = layers.Concat(self.net.conv_a1, self.net.conv_b2, self.net.conv_c2, self.net.conv_d2)



# TODO: rewrite trainer
class Trainer:
    def __init__(self, model, data_reader, num_epochs):
        self.model = model
        self.data_reader = data_reader
        self.num_epochs = num_epochs

    def train(self): # TODO:  make this match our NN and data shape 
        solver = caffe.get_solver(self.get_solver_config())
        for epoch in range(self.num_epochs):
            for batch in self.data_reader.read_data():
                solver.net.blobs['data'].data[...] = batch['data']
                solver.net.blobs['label'].data[...] = batch['label']
                solver.step(1)

    def get_solver_config(self): # TODO: adust to our config
        return {
            'solver_type': 'SGD',
            'base_lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'lr_policy': 'step',
            'gamma': 0.1,
            'stepsize': 100000,
            'max_iter': 50000,
            'snapshot': 5000,
            'snapshot_prefix': 'snapshot',
            'solver_mode': caffe.Solver.GPU
        }

if __name__ == '__main__':
    
    from dataloader import load_MNIST
    caffe.set_mode_cpu()
    
    # loading the data
    x_train, y_train, x_test, y_test = load_MNIST()
    
    # instantiate model;
    model = CVNModel(100, 2) #what does shape and nr of classes correspond to?
    
    train_data, train_labels = model.load_train_data()
    
    solver = model.get_solver()
    
    solver.solve()
    
    # save model, check if '*.caffemodel' syntax is correct
    solver.net.save('trained_model.caffemodel')
    
    
    '''
    Testing
    
    this should go into it's own python script
    '''
    
    # loading the saved model
    net = caffe.Net(model.deploy_prototxt_path, 'trained_model.caffemodel', caffe.TEST)

    # test the model
    test_data, test_labels = model.load_test_data()
    accuracy = model.test(net, test_data, test_labels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    
    
    