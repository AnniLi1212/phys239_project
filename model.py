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
        self.net.data = layers.Input(name='data path', shape=[2, 3, 224, 224])

        self.net.conv1 = layers.Convolution(self.net.data, name='conv1', kernel_size=7, stride=2)
        self.net.pool1 = layers.Pooling(self.net.conv1, name='pool1', kernel_size=3, stride=2)  
        self.net.lrn1 = layers.LRN(self.net.pool1, name='lrn1') #figure out required inputs here
        self.net.conv2 = layers.Convolution(self.net.lrn1,name='conv2', kernel_size=1)
        self.net.conv3 = layers.Convolution(self.net.conv2, name='conv3', kernel_size=3)
        self.net.lrn2 = layers.LRN(self.net.conv3, name='lrn2')
        self.net.pool2 = layers.Pooling(self.net.lrn2, name='pool2', kernel_size=3, stride=2)
        # ADD INCEPTION MODULES -- see figure 1
        self.net.inception1 = None
        self.net.inception2 = None
        self.net.pool3 = layers.Pooling(self.net.inception2, name='pool3', kernel_size = 3, stride=2)
        # inception module
        self.net.inception3 = None
        self.net.pool4 = layers.Pooling(self.net.inception3, name='pool4', kernel_h=6, kernel_w=5, pool=AVE)
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
        self.output = layers.Concat(self.net.conv_a1, self.net.conv_b2, self.net.conv_c2, self.net.conv_d2)
        ## not sure about this!


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
