"""
Created on Tue Feb 28 15:36:24 2023

@author: daniel
"""

from caffe import layers
import numpy as np

#define function that loads data
# ---- ENTER CODE ----


#define main()



# Define the network architecture
# this should match the paper's figure 6


net = caffe.NetSpec()

# adjust input layer to size of event data
net.data = caffe.layers.Input(name='data path', shape=[2, 3, 224, 224])

net.conv1 = layers.Convolution(net.data, name='conv1', kernel_size=7, stride=2)
net.pool1 = layers.Pooling(net.conv1, name='pool1', kernel_size=3, stride=2)  
net.lrn1 = layers.LRN(net.pool1, name='lrn1') #figure out required inputs here
net.conv2 = layers.Convolution(net.lrn1,name='conv2', kernel_size=1)
net.conv3 = layers.Convolution(net.conv2, name='conv3', kernel_size=3)
net.lrn2 = layers.LRN(net.conv3, name='lrn2')
net.pool2 = layers.Pooling(net.lrn2, name='pool2', kernel_size=3, stride=2)
# ADD INCEPTION MODULES -- see figure 1
net.inception1 = None
net.inception2 = None
net.pool3 = layers.Pooling(net.inception2, name='pool3', kernel_size = 3, stride=2)
# inception module
net.inception3 = None
net.pool4 = layers.Pooling(net.inception3, name='pool4', kernel_h=6, kernel_w=5, pool=AVE)
net.softmax = layers.softmax(net.pool4, name='softmax')

net.to_proto() #saves NN settings/weights/training data (?)

#implement multinomial logistic loss

# save training data in file 

# figure out what the output is and how to best return it

