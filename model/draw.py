# Modified from Stanford CS224d PSet2

import sys
import time
import numpy as np

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from copy import deepcopy
import logging
import logging.config

#------------------------------------------------------------------
class Qsampler():
    def __init__(self, input_dim, output_dim, **kwargs):
        """
        """
        self.prior_mean=0.
        self.prior_log_sigma=0.
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.W_mean=tf.Variable(tf.random_uniform([input_dim, output_dim])) 
        self.b_mean=tf.Variable(tf.random_uniform([output_dim]))
        self.W_log=tf.Variable(tf.random_uniform([input_dim, output_dim]))
        self.b_log=tf.Variable(tf.random_uniform([output_dim]))

    def get_dim(self, name):
        if name=='input':
            return self.input_dim
        elif name=='output':
            return self.output_dim
        else:
            raise ValueError

    def sample(self, x, u):
        """
        Return a samples and the corresponding KL term
        Parameters
        --------------
        x:
        
        Returns
        --------------
        z: samples drawn from Q(z|x)
        kl: KL(Q(z|x) || P_z)
        """
        mean=tf.matmul(x, self.W_mean)+self.b_mean
        log_sigma=tf.matmul(x, self.W_log)+self.b_log

        z=mean+tf.exp(log_sigma)*u

        kl=tf.reduce_sum(
                self.prior_log_sigma-log_sigma
                +0.5*(
                    tf.exp(2*log_sigma)+(mean-self.prior_mean)**2
                    )/tf.exp(2*self.prior_log_sigma)
                -0.5
                , 1)

        return z, kl

    def sample_from_prior(self, u):
        """
        Sample z from the prior distribution P_z.
        
        Parameters
        ----------
        u: gaussian random source

        Returns
        -------
        z: samples
        """

        z=self.prior_mean+tf.exp(self.prior_log_sigma)*u

        return z
#------------------------------------------------------------------

class Reader():
    def __init__(self, x_dim, dec_dimi, **kwargs):
        self.x_dim=x_dim
        self.dec_dim=dec_dim
        self.output_dim=2*x_dim

    def get_dim(self, name):
        if name=='input':
            return self.x_dim
        elif name=='output':
            return self.output_dim
        else:
            raise ValueError

    def apply(self, x, x_hat, h_dec):
        return tf.concat(1, [x, x_hat]) 

class AttentionReader():
#todo
    def __init__(self, x_dim, dec_dim, channels, height, width, N, **kwargs):
        self.img_height=height
        self.img_width=width
        self.N=N
        self.x_dim=x_dim
        self.dec_dim=dec_dim
        self.output_dim=2*channels*N*N

#-----------------------------------------------------------------

class Writer():
    def __init__(self, input_dim, output_dim, **kwargs):
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.W=tf.Variable(tf.random_uniform([self.input_dim, self.output_dim]))        
        self.b=tf.Variable(tf.random_uniform([self.output_dim]))

    def transform(h):
       return tf.matmul(h, self.W)+self.b

    def apply(self, h):
        return self.transorm(h)

class AttentionWriter():
#todo
    def __init__(self, input_dim, output_dim, channels, width, height, N, **kwargs):
        self.channels=channels
        self.img_width=width
        self.img_height=height
        self.N=N
        self.input_dim=input_dim
        self.output_dim=output_dim

        assert output_dim==channels*width*height

#-----------------------------------------------------------------

class DrawModel():
#todo
    def __init__(self, n_iter, reader, sampler, writer, x_dim, enc_dim, z_dim, dec_dim, config, **kwargs):
        """
        config should have the following hyper-parameters:
        batch_size
        image_size/x_dim
        image_height, image_width, channel
        n_iter
        enc_dim
        dec_dim
        category_num (which is 10 for cifar-10)

        lr (learning rate)
        """
        self.config=config
        self.add_placeholder()
        self.reader=reader
        self.sampler=sampler
        self.writer=writer
        self.enc_dim=enc_dim
        self.dec_dim=dec_dim
        self.read_dim=reader.get_dim('output')
        self.z_dim=sampler.get_dim('output')
        self.add_encoder_rnn_variable()
        self.add_decoder_rnn_variable()
        self.add_softmax_layer_variable()
        self.build_model()
        self.add_train_op_reconstruct()
        self.add_train_op_classification()
        self.add_predict_op()

    def add_placeholder(self):
        self.x=tf.placeholder(tf.float32, [self.config.batch_size, self.config_image_size])
        self.y=tf.placeholder(tf.int64, [self.config.batch_size, 1])

    def add_encoder_rnn_variable(self):
        self.lstm_cell_encoder=tf.nn.rnn_cell.BasicLSTMCell(self.encoder_dim, forget_bias=1.0)
        self.initial_state_encoder=self.lstm_cell_encoder.zero_state(self.config.batch_size, tf.float32)

    def add_decoder_rnn_variable(self):
        self.lstm_cell_decoder=tf.nn.rnn_cell.BasicLSTMCell(self.decoder_dim, forget_bias=1.0)
        self.initial_state_decoder=self.lstm_cell_decoder.zero_state(self.config.batch_size, tf.float32)

    def add_softmax_variable(self):
        self.W_softmax=tf.Variable(tf.random_uniform([self.encoder_dim+self.z_dim, self.config.category_num]))
        self.b_softmax=tf.Variable(tf.random_uniform([self.config.category_num]))

    def build_model(self):
        """
        generate n_iter steps  
        """
        u=tf.random_normal([self.config.n_iter, self.config.batch_size, self.z_dim])

        state_encoder=self.initial_state_encoder
        state_decoder=self.initial_state_decoder
        x=self.x
        c=tf.zeros([])
        h_enc=tf.zeros([self.config.batch_size, self.enc_dim], tf.float32)
        h_dec=tf.zeros([self.config.batch_size, self.dec_dim], tf.float32)
        x_hat=x-tf.sigmoid(c)
        c=tf.zeros([self.config.batch_size, self.x_dim])

        hidden_enc_list=[]
        hidden_dec_list=[]
        z_list=[]
        kl_list=[]

        for i in xrange(self.n_iter):
            r=self.reader.apply(x, x_hat, h_dec)
            h_enc, state_encoder=self.lstm_cell_encoder(tf.concat(1, [r, h_dec]), state_encoder)
            z, kl=self.sampler.sample(h_enc, u[i, :, :])
            h_dec, state_decoder=self.lstm_cell_decoder(z, state_decoder)
            c=c+self.writer.apply(h_dec)

            hidden_enc_list.append(h_enc)
            hidden_dec_list.append(h_dec)
            z_list.append(z)
            kl_list.append(kl)

        #add reconstruct loss
        x_reconstruct=tf.sigmoid(c)
        loss_reconstruct=tf.reduce_mean(-tf.reduce_sum(self.x*tf.log(c)+(1-self.x)*(1-tf.log(c)), 1))
        kl_all=tf.concat(1, kl_list)
        loss_kl=tf.reduce_mean(kl_all)
        self.loss_reconstruct=loss_kl+loss_reconstruct

        #add classification loss
        softmax_feature_hidden_state=tf.concat(1, hidden_enc_list)
        softmax_feature_z_state=tf.concat(1, z_list)
        softmax_feature=tf.concat(1, [softmax_featrue_hidden_state, softmax_feature_z_state])
        self.logits=tf.matmul(softmax_feature, self.W_softmax)+self.b_softmax
        self.loss_classification=tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.y)

    def add_train_op_reconstruct(self):
        self.train_op_reconstruct=tf.train.AdamOptimizer(self.config.lr).minimize(self.loss_reconstruct)

    def add_train_op_classification(self):
        self.train_op_classification=tf.train.AdamOptimizer(self.config.lr).minimize(self.loss_classification)

    def add_predict_op(self):
        _probability=tf.nn.softmax(self.logits)
        tf.predict=tf.argmax(_probability, 1)

if __name__=='__main__':
#unit test
    q=Qsampler(1,2) 
    q.get_dim('input')
    z,kl=q.sample(tf.constant([[1.]]),tf.constant([[1.,1.]]))
    z=q.sample_from_prior(tf.constant([[1.]]))

