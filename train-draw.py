import tensorflow as tf
import numpy as np
import sys
import time
from utils import utils
import logging
import logging.config
from model.draw import * 

logging.config.fileConfig('logging.conf')
logging.addLevelName(logging.WARNING, "\033[1;34m[%s]\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m[%s]\033[1;0m" % logging.getLevelName(logging.ERROR))

class Config():
    batch_size=64
    image_size=3*32*32
    image_height=32
    image_width=32
    channel=3
    n_iter=10
    enc_dim=256
    dec_dim=256
    z_dim=100
    category_num=10
    lr=0.01
    max_epoch=100

if __name__=='__main__':
    train_x, train_y=utils.load_data('dataset/cifar10/data_batch_1')
    print type(train_x), type(train_y)
    print train_x.shape, train_y.shape
    print train_x
    print np.min(train_y)

    config=Config()
    logging.info('Constructing Qsampler')
    q_sampler=Qsampler(config.enc_dim, config.z_dim)
    logging.info('Construct Qsampler Complete')
    logging.info('Construct Reader')
    reader=Reader(config.image_size, config.dec_dim) 
    logging.info('Construct Reader Complete')
    logging.info('Construct Writer')
    writer=Writer(config.dec_dim, config.image_size)
    logging.info('Construct Writer Complete')
    logging.info('Construct Core Part of DrawModel')
    core=DrawModel(reader, q_sampler, writer, config)
    logging.info('Construct Core Part Complete')

    session=tf.Session() 
    session.run(tf.initialize_all_variables())

    logging.info('Training Start')
    for _epoch in xrange(config.max_epoch):
        logging.info('Epoch {}'.format(_epoch))
        for _step, (_x, _y) in enumerate(data_iterator(train_x, train_y, config.batch_size)):
            logging,info('  step {}'.format(_step))
            session.run([core.train_op_reconstruct, core.train_op_classification], feed_dict={core.x: _x, core.y: _y})
    logging.info('Train Complete')

    test_x, test_y=utils.load_data('dataset/cifar10/test_batch')
    predict=session.run([core.predict], feed_dict={core.x: test_x})
    assert predict.shape[0]==test_y.shape[0], 'rows of predict and label should be equal'
    precision=np.sum(predict==test_y)*1.0/predict.shape[0]
    logging.info('Precision on test set is {}%'.format(precision*100.))
     
    
