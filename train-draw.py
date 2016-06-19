import numpy as np
import tensorflow as tf
import sys
import time
from utils.data_handler import *
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
    enc_dim=64
    dec_dim=64
    z_dim=10
    category_num=10
    lr=0.01
    max_epoch=10
    N=10

if __name__=='__main__':
    train_x, train_y=load_data('dataset/cifar10/data_batch_1')
    train_x=train_x/255.
#    for i in xrange(4):
#        _x, _y=load_data('dataset/cifar10/data_batch_'+str(2+i))
#        _x=_x/255.
#        train_x=np.vstack([train_x, _x])
#        train_y=np.vstack([train_y, _y])

    config=Config()
    logging.info('Constructing Qsampler......')
    q_sampler=Qsampler(config.enc_dim, config.z_dim)
    logging.info('Construct Qsampler Complete!!!')
#    logging.info('Construct Reader......')
#    reader=Reader(config.image_size, config.dec_dim)
#    logging.info('Construct Reader Complete!!!')
    logging.info('Construct AttentionReader......')
    reader=AttentionReader(config.image_size, config.dec_dim, config.channel,
            config.image_height, config.image_width, config.N, config.batch_size)
    logging.info('Construct AttentionReader Complete!!!')
#    logging.info('Construct Writer......')
#    writer=Writer(config.dec_dim, config.image_size)
#    logging.info('Construct Writer Complete!!!')
    logging.info('Construct AttentionWriter......')
    writer=AttentionWriter(config.dec_dim, config.image_size, config.channel,
            config.image_height, config.image_width, config.N, config.batch_size)
    logging.info('Construct AttentionWriter Complete!!!')
    logging.info('Construct Core Part of DrawModel......')
    core=DrawModel(reader, q_sampler, writer, config)
    logging.info('Construct Core Part Complete!!!')

#restore from saver
#    saver=tf.train.Saver()
    session=tf.Session()
    session.run(tf.initialize_all_variables())
#    saver.restore(session, "./tmp/model_35.ckpt")

    logging.info('Training Start......')

    monitor=np.zeros(config.max_epoch)

    for _epoch in xrange(config.max_epoch):
        logging.info('Epoch {}'.format(_epoch))
        for _step, (_x, _y) in enumerate(data_iterator(train_x, train_y, config.batch_size)):
            logging.info('  step {}'.format(_step))
            loss_reconstruct, loss_classification, _=session.run([core.loss_reconstruct, core.loss_classification, core.train_op_classification], feed_dict={core.x: _x, core.y: _y.reshape([-1])})
            logging.info('  loss_reconstruct: {}, loss_classification: {}'.format(loss_reconstruct, loss_classification))
            monitor[_epoch]+=loss_classification
#        save_path=saver.save(session, './tmp/model_'+str(_epoch)+'.ckpt')
#        logging.info('Model saved in file: {}'.format(save_path))
        logging.info('Total classification loss is: {}'.format(monitor[_epoch]))

    logging.info('Train Complete!!!')


    logging.info('Testing Start......')
    test_x, test_y=load_data('dataset/cifar10/test_batch')
    predict=np.array([])
    label=np.array([])
    for _step, (_x, _y) in enumerate(data_iterator(test_x, test_y, config.batch_size)):
        if _step==0:
            predict=np.reshape(session.run(core.predict, feed_dict={core.x: _x}), [-1, 1])
            label=_y
        else:
            predict=np.vstack([predict, np.reshape(session.run(core.predict, feed_dict={core.x: _x}), [-1, 1])])
            label=np.vstack([label, _y])
        logging.info('have predicted {}/{} samples already...'.format(predict.shape[0], test_y.shape[0]))

    assert predict.shape[0]==label.shape[0], 'rows of predict and label should be equal'
    precision=np.sum(predict==label)*1.0/predict.shape[0]
    logging.info('Precision on test set is {}%'.format(precision*100.))

    for i in xrange(config.max_epoch):
        print monitor[i]

