#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 14:31:07 2023

@author: daniel
"""
import uproot
import numpy as np
import tensorflow as tf

def load_data(file_path, tree_name, image_branch_name, label_branch_name):
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        images = np.stack(tree.arrays(image_branch_name, library="np"))
        labels = tree.arrays(label_branch_name, library="np")
    return images, labels

def load_MNIST():
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist

    # Split the dataset into training and testing sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Create Dataset objects for the training and testing sets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # Take a subset of the data for code testing purposes
    num_train_samples = 100
    num_test_samples = 20
    train_subset = train_dataset.take(num_train_samples)
    test_subset = test_dataset.take(num_test_samples)

    # Convert the subsets to NumPy arrays
    X_train_small, y_train_small = zip(*train_subset.as_numpy_iterator())
    X_test_small, y_test_small = zip(*test_subset.as_numpy_iterator())
    X_train_small = np.array(X_train_small)
    y_train_small = np.array(y_train_small)
    X_test_small = np.array(X_test_small)
    y_test_small = np.array(y_test_small)

    
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    
    file_path = 'practice_train_5k.root'
    
    #images, labels = load_data(practice_train_5k.root)
    
    #print(images)
    image_data = []
    '''
    
    with uproot.open(file_path) as file:
        for tree_name in file.keys():
            tree = file[tree_name]
            for branch_name in tree.keys():
                branch = tree[branch_name]
                #print(branch_name)
                # Check if the branch has an array-like structure
                if isinstance(branch.interpretation, uproot.interpretation.jagged.JaggedArray):
                    try:
                        # Attempt to load branch data as an image (assuming it's an array of arrays)
                        images = np.stack(tree.arrays(branch_name, library="np"))
                        print(tree.arrays(branch_name, library='np'))
                        # Append image data to the list
                        image_data.append((tree_name.decode('utf-8'), branch_name.decode('utf-8'), images))
                    except ValueError:
                        # Branch data is not in the expected image format, skip this branch
                        print('Branch data is not in the expected image format')
                        continue

    print(image_data)
    '''
    
    x, y, _, _ = load_MNIST()
    
    print(x)
