from caffe import layers
import numpy as np
from model import DataReader, CVNModel, Inception_Module, Trainer

def main():
    root_file = "" #need to specify
    batch_size = #need to specify
    data_reader = DataReader(root_file, batch_size)

    # model architechture
    input_shape = (2,3,224,224) #need to double check
    num_classes = 3
    model = CVNModel(input_shape, num_classes)
    model.create_network()
    
    inception1 = Inception_Module(model.net.pool2, "inception1")
    model.net.inception1 = inception1.output

    inception2 = Inception_Module(inception1, "inception2")
    model.net.inception2 = inception2.output

    inception3 = Inception_Module(model.net.pool3, "inception3")
    model.net.inception3 = inception3.output

    model.net.pool4 = layers.Pooling(model.net.inception3, name='pool4', 
                                     kernel_h=6, kernel_w=5, pool=AVE)
    model.net.softmax = layers.softmax(model.net.pool4, name='softmax')

    num_epochs = #need to determine
    trainer = Trainer(model, data_reader, num_epochs)
    trainer.train()

if __name__ == "__main__":
    main()

# save training data in file 

# figure out what the output is and how to best return it

