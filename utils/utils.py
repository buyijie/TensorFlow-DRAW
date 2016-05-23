import numpy as np
import tensorflow as tf
import cPickle as pickle
import sys

def LoadData (dataset_name):
    """
    dataset_name: file_path
    return feature, label, type is ndarray 
    """
    with open(dataset_name, 'rb') as f:
        data=pickle.load(f)
    return data['data'], np.array(data['labels'], dtype=np.int)
             

if __name__=="__main__":
    train_x, train_y=LoadData("../dataset/cifar10/data_batch_1")
