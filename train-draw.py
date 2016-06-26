import numpy as np
import tensorflow as tf
import sys
import time
from utils.data_handler import *
import logging
import logging.config
from model.draw import *
import pickle

logging.config.fileConfig('logging.conf')
logging.addLevelName(logging.WARNING, "\033[1;34m[%s]\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m[%s]\033[1;0m" % logging.getLevelName(logging.ERROR))

class Config():
    batch_size=50
    image_size=3*32*32
    image_height=32
    image_width=32
    channel=3
    n_iter=10
    enc_dim=128
    dec_dim=128
    z_dim=25
    category_num=10
    lr=0.01
    max_epoch=20
    N=10

if __name__=='__main__':
    train_x, train_y=load_data('dataset/cifar10/data_batch_1')
    train_x=train_x/255.
#    train_x=train_x[:50, :]
#    train_y=train_y[:50, :]
#    for i in xrange(4):
#        _x, _y=load_data('dataset/cifar10/data_batch_'+str(2+i))
#        _x=_x/255.
#        train_x=np.vstack([train_x, _x])
#        train_y=np.vstack([train_y, _y])

    assert train_x.shape[0]==train_y.shape[0], 'size of x and y must be equal'

    logging.info('Total data size is {}'.format(train_x.shape[0]))

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
    saver=tf.train.Saver()
    session=tf.Session()
    session.run(tf.initialize_all_variables())

    logging.info('Training Start......')

    monitor={
        'loss_reconstruct_with_kl': [],
        'loss_reconstruct': [],
        'loss_kl': [],
        'loss_classification': [],
        'classification_precision': []
            }

    for _epoch in xrange(config.max_epoch):
        logging.info('Epoch {}'.format(_epoch))

        #add record of current epoch
        for _key, _val in monitor.items():
            monitor[_key].append(0.)

        for _step, (_x, _y) in enumerate(data_iterator(train_x, train_y, config.batch_size)):
            logging.info('  step {}'.format(_step))

            #one step
            (_tmp_loss, _tmp_loss_kl, loss_reconstruct, loss_classification,
                    _)=session.run([core._loss_reconstruct, core._loss_kl,
                        core.loss_reconstruct, core.loss_classification,
                        core.train_op_reconstruct], feed_dict={core.x: _x,
                            core.y: _y.reshape([-1])})

            logging.info('  loss_reconstruct: {}, loss_classification: {}'.format(loss_reconstruct, loss_classification))
            logging.info('  _loss_reconstruct: {}, loss_kl: {}'.format(_tmp_loss, _tmp_loss_kl))

            monitor['loss_reconstruct_with_kl'][-1]+=loss_reconstruct
            monitor['loss_reconstruct'][-1]+=_tmp_loss
            monitor['loss_kl'][-1]+=_tmp_loss_kl
            monitor['loss_classification'][-1]+=loss_classification

        save_path=saver.save(session, './tmp/reconstruct/model_'+str(_epoch)+'.ckpt')
        logging.info('Model saved in file: {}'.format(save_path))

        for _key, _val in monitor.items():
            monitor[_key][-1]/=(train_x.shape[0]/config.batch_size)

        logging.info('Average reconstruct loss: {}, classification loss: {}'.format(
            monitor['loss_reconstruct_with_kl'][-1], monitor['loss_classification'][-1]))

        pickle.dump(monitor, open('./tmp/reconstruct/monitor_'+str(_epoch)+'.pkl', 'wb'))
        with open('./tmp/reconstruct/monitor_'+str(_epoch)+'.log', 'w') as out:
            for _key, _val in monitor.items():
                out.write(_key+'\n')
                for _record in _val:
                    out.write(str(_record)+' ')
                out.write('\n')

    logging.info('Train Reconstruct Complete!!!')
    logging.info('Train Classification Start...')

    for _epoch in xrange(config.max_epoch):
        logging.info('Epoch {}'.format(_epoch))

        #add record of current epoch
        for _key, _val in monitor.items():
            monitor[_key].append(0.)

        for _step, (_x, _y) in enumerate(data_iterator(train_x, train_y, config.batch_size)):
            logging.info('  step {}'.format(_step))

            #one step
            (_tmp_loss, _tmp_loss_kl, loss_reconstruct, loss_classification,
                    _)=session.run([core._loss_reconstruct, core._loss_kl,
                        core.loss_reconstruct, core.loss_classification,
                        core.train_op_classification], feed_dict={core.x: _x,
                            core.y: _y.reshape([-1])})

            logging.info('  loss_reconstruct: {}, loss_classification: {}'.format(loss_reconstruct, loss_classification))
            logging.info('  _loss_reconstruct: {}, loss_kl: {}'.format(_tmp_loss, _tmp_loss_kl))

            monitor['loss_reconstruct_with_kl'][-1]+=loss_reconstruct
            monitor['loss_reconstruct'][-1]+=_tmp_loss
            monitor['loss_kl'][-1]+=_tmp_loss_kl
            monitor['loss_classification'][-1]+=loss_classification

        save_path=saver.save(session, './tmp/classification/model_'+str(_epoch)+'.ckpt')
        logging.info('Model saved in file: {}'.format(save_path))

        for _key, _val in monitor.items():
            monitor[_key][-1]/=(train_x.shape[0]/config.batch_size)

        logging.info('Average reconstruct loss: {}, classification loss: {}'.format(
            monitor['loss_reconstruct_with_kl'][-1], monitor['loss_classification'][-1]))

        pickle.dump(monitor, open('./tmp/classification/monitor_'+str(_epoch)+'.pkl', 'wb'))
        with open('./tmp/classification/monitor_'+str(_epoch)+'.log', 'w') as out:
            for _key, _val in monitor.items():
                out.write(_key+'\n')
                for _record in _val:
                    out.write(str(_record)+' ')
                out.write('\n')

    logging.info('Train Classification Complete!!!')

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
    monitor['classification_precision'].append(precision)
    logging.info('Precision on test set is {}%'.format(precision*100.))

