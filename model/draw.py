# Modified from Stanford CS224d PSet2

import sys
import time
import numpy as np

import tensorflow as tf
from copy import deepcopy

def data_iterator(data, batch_size):
	'''
	data contain both x, y
	'''
	yield(x,y)

class Config():
	batch_size=64
	hidden_size=100
	max_epochs=16
	early_stopping=2
	dropout=0.1
	lr=0.001

class Model():

	def load_data(self):
		'''
		self.train=
		self.valid=
		self.test=
		'''

	def add_placeholders(self):
		'''
		self.input_placeholder=
		self.labels_placeholder=
		self.droput_placeholder=
		'''

	def add_model(self,inputs):
		return output
	
	def add_loss_op(self, output):
		return loss

	def add_training_op(self, loss):
		return train_op
	
	def __init__(self, config):
		self.config=config
		self.load_data()
		self.add_placeholders()
		y=self.add_model()
		self.loss=self.add_loss_op(y)
		self.train_step=self.add_training_op(self.loss)
	
	def run_epoch(self, session, data, train_op=None, verbose=10):
		config=self.config
		dp=config.dropout
		if not train_op:
			train_op=tf.no_op()
			dp=1
			
		total_steps=sum(1 for x in data_iterator(data, config.batch_size))
		total_loss=[]
		# for rnn
		#state=self.initial_state.eval()
		for step, (x, y) in enumerate(
				data_iterator(data, config.batch_size)):
			feed={self.input_placeholder: x,
					self.labels_placeholder: y,
					#self.initial_state: state, # for rnn
					self.dropout_placeholder: dp}
			loss, state, _ = session.run(
					[self.loss, self.final_state, train_op], feed_dict=feed)
			total_loss.append(loss)
			if verbose and step % verbose == 0:
				sys.stdout.write('\r{} / {} : loss = {}'.format(
					step, total_steps, np.mean(total_loss)))
				sys.stdout.flush()
		if verbose:
			sys.stdout.write('\r')
		return loss

def test(train_x, train_y, validate_x, validate_y, test_x, test_y=None):
	config=Config()
	config2=deepcopy(config)
	with tf.variable_scope('NN') as scope:
		model1=Model(config)
		scope.reuse_variables()
		model2=Model(config2)
	
	init=tf.initialize_all_variables()
	saver=tf.train.Saver()

	with tf.Session() as session:
		best_val_loss=float('inf')
		best_val_epoch=0

		session.run(init)
		for epoch in xrange(config.max_epochs):
			print 'Epoch {}'.format(epoch)
			start=time.time()

			train_loss=model.run_epoch(
					session, model.train, train_op=model.train_step)
			valid_loss=model.run_epoch(session, model.valid)
			print 'Training loss: {}'.format(train_loss)
			print 'Validation loss: {}'.format(valid_loss)
			if valid_loss<best_val_loss:
				best_val_loss=valid_loss
				best_val_epoch=epoch
				saver.save(session, './nn.weights')
			if epoch-best_val_epoch>config.early_stopping:
				break
			print 'Total time: {}'.format(time.time()-start)

		saver.restore(session, 'nn.weights')
		test_loss=model.run_epoch(session, model.test)
		print '=-='*5
		print 'Test loss: {}'.format(test_loss)
		print '=-='*5

if __name__=="__main__":
	test()
