import numpy as np
import tensorflow as tf
import cPickle as pickle
import sys

def load_data (dataset_name):
    """
    dataset_name: file_path
    return feature, label, type is ndarray 
    """
    with open(dataset_name, 'rb') as f:
        data=pickle.load(f)
    return data['data'].astype(np.float32), np.array(data['labels'], dtype=np.int64).reshape(-1, 1)

def data_iterator(data_x, data_y, batch_size):
    """
    """
    assert data_x.shape[0]==data_y.shape[0], 'rows of x and y must be equal'
    length=data_x.shape[0]
    batch_length=length//batch_size
    for i in xrange(batch_length):
        x=data_x[i*batch_size:(i+1)*batch_size, :]
        y=data_y[i*batch_size:(i+1)*batch_size, :]
        yield(x, y)

    x=data_x[batch_length*batch_size:, :]
    y=data_y[batch_length*batch_size:, :]
    yield(x, y)

if __name__=="__main__":
    train_x, train_y=load_data("../dataset/cifar10/data_batch_1")
    for step, (x, y) in enumerate(data_iterator(train_x, train_y, 64)):
        print step
        print x
        print y
