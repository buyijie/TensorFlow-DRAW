# take the framework from jbornschein/draw
import sys
import time
import numpy as np

import tensorflow as tf
from copy import deepcopy
import logging
import logging.config

def my_batched_dot(A, B):
    """
    Batched version of dot-product.
    for A[dim_1, dim_2, dim_3] and B[dim_1, dim_3, dim_4]
    return C[dim_1, dim_2, dim_4]
    """
    C=tf.tile(tf.expand_dims(A, 3), tf.pack([1, 1, 1, tf.shape(B)[2]])) \
        *tf.tile(tf.expand_dims(B, 1), tf.pack([1, tf.shape(A)[1], 1, 1]))
    return tf.reduce_sum(C, 2)

class ZoomableAttentionWindow():
    def __init__(self, channels, img_height, img_width, N, batch_size):
        """
        A zoomable attention window for images
        """
        self.channels=channels
        self.img_height=img_height
        self.img_width=img_width
        self.N=N
        self.batch_size=batch_size

    def filterbank_matrices(self, center_y, center_x, delta, sigma):
        """
        create a Fy and a Fx
        shape Fy (batch_size X N X img_height)
        shape Fx (batch_size X N X img_width)
        """
        tol=1e-4
        N=self.N

        rng=np.arange(N, dtype=np.float32)+N/2.+0.5

        muX=tf.expand_dims(center_x, 1)+tf.expand_dims(delta, 1)*rng
        muY=tf.expand_dims(center_y, 1)+tf.expand_dims(delta, 1)*rng

        a=np.arange(self.img_width, dtype=np.float32)
        b=np.arange(self.img_height, dtype=np.float32)

        FX=tf.exp(-(a-tf.expand_dims(muX, 2))**2/2./tf.expand_dims(tf.expand_dims(sigma, 1), 2)**2)
        FY=tf.exp(-(b-tf.expand_dims(muY, 2))**2/2./tf.expand_dims(tf.expand_dims(sigma, 1), 2)**2)
        FX=FX/(tf.expand_dims(tf.reduce_sum(FX, 2), 2)+tol)
        FY=FY/(tf.expand_dims(tf.reduce_sum(FY, 2), 2)+tol)

        return FY, FX

    def read(self, images, center_y, center_x, delta, sigma):
        """
        Extract a batch of attention windows from the given images.

        return windows of shape (batch_size X N**2)
        """
        N=self.N
        channels=self.channels

        # Reshape input into proper 2d images
        I=tf.reshape(images, [self.batch_size*channels, self.img_height, self.img_width])

        # Get separable filterbank
        FX, FY=self.filterbank_matrices(center_y, center_x, delta, sigma)

        FY=tf.reshape(tf.tile(tf.reshape(FY, [self.batch_size, -1]), [1, channels]),
                [self.batch_size*channels, self.N, self.img_height])
        FX=tf.reshape(tf.tile(tf.reshape(FX, [self.batch_size, -1]), [1, channels]),
                [self.batch_size*channels, self.N, self.img_width])
        W=my_batched_dot(my_batched_dot(FY, I), tf.transpose(FY, perm=[0, 2, 1]))
        return tf.reshape(W, [self.batch_size, channels*N*N])

    def write(self, windows, center_y, center_x, delta, sigma):
        """
        Write a batch of windows into full sized images.
        """
        N=self.N
        channels=self.channels

        W=tf.reshape(windows, [self.batch_size*channels, N, N])

        FY, FX=self.filterbank_matrices(center_y, center_x, delta, sigma)
        FY=tf.reshape(tf.tile(tf.reshape(FY, [self.batch_size, -1]), [1, channels]),
                [self.batch_size*channels, self.N, self.img_height])
        FX=tf.reshape(tf.tile(tf.reshape(FX, [self.batch_size, -1]), [1, channels]),
                [self.batch_size*channels, self.N, self.img_width])

        I=my_batched_dot(my_batched_dot(tf.transpose(FY, perm=[0, 2, 1]), W), FX)

        return tf.reshape(I, [self.batch_size, channels*self.img_height*self.img_width])

    def nn2att(self, l):
        """
        Convert neural-net outputs to attention parameters
        Input shape (batch_size X 5)
        Returns:
        center_y
        center_x
        delta
        sigma
        gamma
        """
        center_y=l[:, 0]
        center_x=l[:, 1]
        log_delta=l[:, 2]
        log_sigma=l[:, 3]
        log_gamma=l[:, 4]

        delta=tf.exp(log_delta)
        sigma=tf.exp(log_sigma/2.)
        gamma=tf.expand_dims(tf.exp(log_gamma), 1)

        center_x=(center_x+1.)/2.*self.img_width
        center_y=(center_y+1.)/2.*self.img_height
        delta=(max(self.img_width, self.img_height)-1)/(self.N-1)*delta

        return center_y, center_x, delta, sigma, gamma

