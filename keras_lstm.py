from __future__ import print_function
import argparse
import logging
import numpy as np
from time import time
import sys
import utils as U
import pickle as pk

logger = logging.getLogger(__name__)

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train", dest="train_path", type=str, metavar='<str>', required=True, help="The path to the training set")
parser.add_argument("-tu", "--tune", dest="dev_path", type=str, metavar='<str>', required=True, help="The path to the development set")
parser.add_argument("-ts", "--test", dest="test_path", type=str, metavar='<str>', required=True, help="The path to the test set")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-t", "--model-type", dest="model_type", type=str, metavar='<str>', default='cnnmeanp', help="Model type (cnnmeanp) (default=cnnmeanp|cnnwang2016)")
parser.add_argument("-u", "--rec-unit", dest="recurrent_unit", type=str, metavar='<str>', default='lstm', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=50, help="Embeddings dimension (default=300)")
parser.add_argument("-c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=300, help="CNN output dimension. '0' means no CNN layer (default=0)")
parser.add_argument("-w", "--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=2, help="CNN window size. (default=2)")
parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=0, help="RNN dimension. '0' means no RNN layer (default=0)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size for training (default=32)")
parser.add_argument("-be", "--batch-size-eval", dest="batch_size_eval", type=int, metavar='<int>', default=256, help="Batch size for evaluation (default=256)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=30000, help="Vocab size (default=30000)")
parser.add_argument("--coef", dest="activation_coef", type=float, metavar='<int>', default=1.0, help="The last layers sharpness coefficient (default=1.0)")
parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>', help="(Optional) The path to the existing vocab file (*.pkl)")
parser.add_argument("--skip-init-bias", dest="skip_init_bias", action='store_true', help="Skip initialization of the last layer bias")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file (Word2Vec format)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=50, help="Number of epochs (default=50)")
parser.add_argument("--maxl", dest="maxl", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")

# Needed for resuming current model/epoch
parser.add_argument("--arch-path", dest="arch_path", type=str, metavar='<str>', help="The path to the model architecture file (*.json)")
parser.add_argument("--model-path", dest="model_path", type=str, metavar='<str>', help="The path to the model weights file (*.h5)")
parser.add_argument("--continue", dest="is_continue_model", action='store_true', help="Flag to use existing model and continue training")

args = parser.parse_args()

train_path = args.train_path
dev_path = args.dev_path
test_path = args.test_path
out_dir = args.out_dir_path
model_type = args.model_type
algorithm = args.algorithm
recurrent_unit = args.recurrent_unit
emb_dim = args.emb_dim
cnn_dim = args.cnn_dim
cnn_window_size = args.cnn_window_size
rnn_dim = args.rnn_dim
activation_coef = args.activation_coef
batch_size = args.batch_size
batch_size_eval = args.batch_size_eval
vocab_size = args.vocab_size
vocab_path = args.vocab_path
skip_init_bias = args.skip_init_bias
emb_path = args.emb_path
nb_epoch = args.epochs
maxlen = args.maxl
seed = args.seed

arch_path = args.arch_path
model_path = args.model_path
is_continue_model = args.is_continue_model

U.mkdir_p(out_dir + '/preds')
U.mkdir_p(out_dir + '/models')
U.set_logger(out_dir)
U.print_args(args)

valid_model_type = {
	'cnnmeanp',
	'cnnwang2016'
}

assert model_type in valid_model_type
assert algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
assert recurrent_unit in {'lstm', 'gru', 'simple'}

###############################################################################################################################
## Import Keras related modules after setting the seed
#

if seed > 0:
	logger.info('Setting np.random.seed(%d) before importing keras' % seed)
	np.random.seed(seed)

import keras.optimizers as opt
import keras.backend as K
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
#from keras.layers.convolutional import Convolution1D
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.utils.visualize_util import plot
from keras.utils import np_utils
from my_layers import MeanOverTime, MulConstant, Conv1DWithMasking
from sklearn.metrics import accuracy_score
from w2vEmbReader import W2VEmbReader as EmbReader
import asap_reader as asap
from Evaluator import Evaluator

###############################################################################################################################
## Prepare data
#
# data_x is a list of lists
(train_qn_x, train_ans_x, train_y), (dev_qn_x, dev_ans_x, dev_y), (test_qn_x, test_ans_x, test_y), vocab, vocab_size, overal_maxlen = asap.get_data(
	(train_path, dev_path, test_path), vocab_size, maxlen, tokenize_text=False, to_lower=True, sort_by_len=False, vocab_path=vocab_path)

# Dump vocab
with open(out_dir + '/vocab.pkl', 'wb') as vocab_file:
	pk.dump(vocab, vocab_file)

# Pad sequences for mini-batch processing

train_qn_x = sequence.pad_sequences(train_qn_x, maxlen=overal_maxlen)
train_ans_x = sequence.pad_sequences(train_ans_x, maxlen=overal_maxlen)
dev_qn_x = sequence.pad_sequences(dev_qn_x, maxlen=overal_maxlen)
dev_ans_x = sequence.pad_sequences(dev_ans_x, maxlen=overal_maxlen)
test_qn_x = sequence.pad_sequences(test_qn_x, maxlen=overal_maxlen)
test_ans_x = sequence.pad_sequences(test_ans_x, maxlen=overal_maxlen)

###############################################################################################################################
## Some statistics
#

bincount = np.bincount(train_y)
most_frequent_class = bincount.argmax()
np.savetxt(out_dir + '/bincount.txt', bincount, fmt='%i')

train_y = np.array(train_y, dtype=K.floatx())
dev_y = np.array(dev_y, dtype=K.floatx())
test_y = np.array(test_y, dtype=K.floatx())

train_mean = train_y.mean()
train_std = train_y.std()
dev_mean = dev_y.mean()
dev_std = dev_y.std()
test_mean = test_y.mean()
test_std = test_y.std()

logger.info('Statistics:')

logger.info('  train_qn_x shape: ' + str(np.array(train_qn_x).shape))
logger.info('  dev_qn_x shape:   ' + str(np.array(dev_qn_x).shape))
logger.info('  test_qn_x shape:  ' + str(np.array(test_qn_x).shape))

logger.info('  train_ans_x shape: ' + str(np.array(train_ans_x).shape))
logger.info('  dev_ans_x shape:   ' + str(np.array(dev_ans_x).shape))
logger.info('  test_ans_x shape:  ' + str(np.array(test_ans_x).shape))

logger.info('  train_y shape: ' + str(train_y.shape))
logger.info('  dev_y shape:   ' + str(dev_y.shape))
logger.info('  test_y shape:  ' + str(test_y.shape))

logger.info('  train_y mean: %.3f, stdev: %.3f, MFC: %i' % (train_mean, train_std, most_frequent_class))

###############################################################################################################################
## Optimizer algorithm
#

clipvalue = 0
clipnorm = 10

if algorithm == 'rmsprop':
	optimizer = opt.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)	
elif algorithm == 'sgd':
	optimizer = opt.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False, clipnorm=clipnorm, clipvalue=clipvalue)
elif algorithm == 'adagrad':
	optimizer = opt.Adagrad(lr=0.01, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
elif algorithm == 'adadelta':
	optimizer = opt.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
elif algorithm == 'adam':
	optimizer = opt.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm, clipvalue=clipvalue)
elif algorithm == 'adamax':
	optimizer = opt.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm, clipvalue=clipvalue)

###############################################################################################################################
## Loss and metric
#

loss = 'binary_crossentropy'
metric = 'accuracy'

###############################################################################################################################
## Recurrence unit type
#

if recurrent_unit == 'lstm':
	from keras.layers.recurrent import LSTM as RNN
elif recurrent_unit == 'gru':
	from keras.layers.recurrent import GRU as RNN
elif recurrent_unit == 'simple':
	from keras.layers.recurrent import SimpleRNN as RNN

###############################################################################################################################
## Building model
#

if model_type == 'cnnmeanp':
	logger.info('Building a CNN model with MeanOverTime (Mean Pooling)')
	from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
	
	cnn_border_mode='same'
	
	sequenceQn = Input(shape=(overal_maxlen,), dtype='int32')
	sequenceAns = Input(shape=(overal_maxlen,), dtype='int32')
	outputQn = Embedding(vocab_size, emb_dim, mask_zero=True, name='QnEmbedding')(sequenceQn)
	outputAns = Embedding(vocab_size, emb_dim, mask_zero=True, name='AnsEmbedding')(sequenceAns)

	if cnn_dim > 0:
		outputQn = Conv1DWithMasking(nb_filter=cnn_dim, filter_length=cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(outputQn)
		outputAns = Conv1DWithMasking(nb_filter=cnn_dim, filter_length=cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(outputAns)
	
	outputMeanQn = MeanOverTime(mask_zero=True)(outputQn)
	outputMeanAns = MeanOverTime(mask_zero=True)(outputAns)
	
	merged = merge([outputMeanQn, outputMeanAns], mode='concat', concat_axis=-1)

	densed = Dense(1)(merged)
	score = Activation('sigmoid')(densed)
	model = Model(input=[sequenceQn,sequenceAns], output=score)
	
	# get the WordEmbedding layer index
	model.emb_index = 0
	model_layer_index = 0
	for test in model.layers:
		if (test.name == 'QnEmbedding' or test.name == 'AnsEmbedding'):
			model.emb_index = model_layer_index
			# Initialize embeddings if requested
			if emb_path:
				logger.info('Initializing lookup table')
				emb_reader = EmbReader(emb_path, emb_dim=emb_dim)
				model.layers[model.emb_index].W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].W.get_value()))
			
		model_layer_index += 1	

if model_type == 'cnnwang2016':
	logger.info('Building a CNN model (Zhiguo Wang, 2016) with S+,S-,T+,T- as input, and MaxPooling)')
	from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, pooling

	assert cnn_dim > 0
	
	cnn_border_mode='same'
	
	sequenceSplus = Input(shape=(overal_maxlen,), dtype='int32')
	outputSplus = Embedding(vocab_size, emb_dim, mask_zero=True, name='SplusEmbedding')(sequenceSplus)
	convSplus1 = Conv1DWithMasking(nb_filter=cnn_dim, filter_length=1, border_mode=cnn_border_mode, subsample_length=1)(outputSplus)
	convSplus2 = Conv1DWithMasking(nb_filter=cnn_dim, filter_length=2, border_mode=cnn_border_mode, subsample_length=1)(outputSplus)
	convSplus3 = Conv1DWithMasking(nb_filter=cnn_dim, filter_length=3, border_mode=cnn_border_mode, subsample_length=1)(outputSplus)

	sequenceSminus = Input(shape=(overal_maxlen,), dtype='int32')
	outputSminus = Embedding(vocab_size, emb_dim, mask_zero=True, name='SminusEmbedding')(sequenceSminus)
	convSminus1 = Conv1DWithMasking(nb_filter=cnn_dim, filter_length=1, border_mode=cnn_border_mode, subsample_length=1)(outputSminus)
	convSminus2 = Conv1DWithMasking(nb_filter=cnn_dim, filter_length=2, border_mode=cnn_border_mode, subsample_length=1)(outputSminus)
	convSminus3 = Conv1DWithMasking(nb_filter=cnn_dim, filter_length=3, border_mode=cnn_border_mode, subsample_length=1)(outputSminus)

	sequenceTplus = Input(shape=(overal_maxlen,), dtype='int32')
	outputTplus = Embedding(vocab_size, emb_dim, mask_zero=True, name='TplusEmbedding')(sequenceTplus)
	convTplus1 = Conv1DWithMasking(nb_filter=cnn_dim, filter_length=1, border_mode=cnn_border_mode, subsample_length=1)(outputTplus)
	convTplus2 = Conv1DWithMasking(nb_filter=cnn_dim, filter_length=2, border_mode=cnn_border_mode, subsample_length=1)(outputTplus)
	convTplus3 = Conv1DWithMasking(nb_filter=cnn_dim, filter_length=3, border_mode=cnn_border_mode, subsample_length=1)(outputTplus)

	sequenceTminus = Input(shape=(overal_maxlen,), dtype='int32')
	outputTminus = Embedding(vocab_size, emb_dim, mask_zero=True, name='TminusEmbedding')(sequenceTminus)
	convTminus1 = Conv1DWithMasking(nb_filter=cnn_dim, filter_length=1, border_mode=cnn_border_mode, subsample_length=1)(outputTminus)
	convTminus2 = Conv1DWithMasking(nb_filter=cnn_dim, filter_length=2, border_mode=cnn_border_mode, subsample_length=1)(outputTminus)
	convTminus3 = Conv1DWithMasking(nb_filter=cnn_dim, filter_length=3, border_mode=cnn_border_mode, subsample_length=1)(outputTminus)

	mergedS1 = merge([convSplus1, convSminus1], mode='sum', concat_axis=-1)
	mergedS2 = merge([convSplus2, convSminus2], mode='sum', concat_axis=-1)
	mergedS3 = merge([convSplus3, convSminus3], mode='sum', concat_axis=-1)
	mergedS  = merge([mergedS1, mergedS2, mergedS3], mode='concat', concat_axis=-1)
	mergedStanh = Activation('tanh')(mergedS)
	maxPoolS = MeanOverTime()(mergedStanh) # PLEASE REMEMBER TO CHANGE THIS TO MAX POOLING

	mergedT1 = merge([convTplus1, convTminus1], mode='sum', concat_axis=-1)
	mergedT2 = merge([convTplus2, convTminus2], mode='sum', concat_axis=-1)
	mergedT3 = merge([convTplus3, convTminus3], mode='sum', concat_axis=-1)
	mergedT  = merge([mergedT1, mergedT2, mergedT3], mode='concat', concat_axis=-1)
	mergedTtanh = Activation('tanh')(mergedT)
	maxPoolT = MeanOverTime()(mergedTtanh) # PLEASE REMEMBER TO CHANGE THIS TO MAX POOLING

	combinedOutput = merge([maxPoolS, maxPoolT], mode='concat', concat_axis=-1)
	densed = Dense(1)(combinedOutput)
	score = Activation('sigmoid')(densed)

	model = Model(input=[sequenceSplus,sequenceSminus,sequenceTplus,sequenceTminus], output=score)
	
	# get the WordEmbedding layer index
	if emb_path:
		logger.info('Initializing lookup table')
		emb_reader = EmbReader(emb_path, emb_dim=emb_dim)

		model_layer_index = 0
		for test in model.layers:
			if (test.name == 'SplusEmbedding' or test.name == 'SminusEmbedding' or test.name == 'TplusEmbedding' or test.name == 'TminusEmbedding'):
				model.layers[model_layer_index].W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model_layer_index].W.get_value()))
			model_layer_index += 1	


model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

plot(model, to_file = out_dir + '/model.png')

logger.info('  Done')

###############################################################################################################################
## Save model architecture
#

logger.info('Saving model architecture')
with open(out_dir + '/model_arch.json', 'w') as arch:
	arch.write(model.to_json(indent=2))

logger.info('---------------------------------------------------------------------------------------')
	
###############################################################################################################################
## Training
#

logger.info('Initial Evaluation:')
evl = Evaluator(logger, out_dir, (train_qn_x, train_ans_x, train_y), (dev_qn_x, dev_ans_x, dev_y) , (test_qn_x, test_ans_x, test_y), model_type, batch_size_eval=batch_size_eval, print_info=True)
evl.evaluate(model, -1)

evl.print_info()

total_train_time = 0
total_eval_time = 0

for ii in range(nb_epoch):
	# Training
	train_input = [train_qn_x, train_ans_x]
	if model_type == 'cnnwang2016':
		train_input = [train_qn_x, train_qn_x, train_ans_x, train_ans_x]

	t0 = time()
	# this model.fit function is the neuralnet training
	train_history = model.fit(train_input, train_y, batch_size=batch_size, nb_epoch=1, verbose=0)

	tr_time = time() - t0
	tr_time_min = tr_time/60
	total_train_time += tr_time

	# Save model for every epoch
	model.save_weights(out_dir + '/models/model_weights_epoch'+ str(ii) +'.h5', overwrite=True)
	
	# Evaluate
	t0 = time()
	evl.evaluate(model, ii)
	evl_time = time() - t0
	evl_time_min = evl_time/60
	total_eval_time += evl_time
	
	# Print information
	train_loss = train_history.history['loss'][0]
	if metric == 'accuracy':
		train_metric = train_history.history['acc'][0]
	else:
		train_metric = train_history.history[metric][0]
	logger.info('Epoch %d, train: %is (%.1fm), evaluation: %is (%.1fm)' % (ii, tr_time, tr_time_min, evl_time, evl_time_min))
	logger.info('[Train] loss: %.4f, metric: %.4f' % (train_loss, train_metric))

	evl.print_info()


###############################################################################################################################
## Summary of the results
#

total_time = total_train_time + total_eval_time

total_train_time_hours = total_train_time/3600
total_eval_time_hours = total_eval_time/3600
total_time_hours = total_time/3600

logger.info('Training:   %i seconds in total (%.1f hours)' % (total_train_time, total_train_time_hours))
logger.info('Evaluation: %i seconds in total (%.1f hours)' % (total_eval_time, total_eval_time_hours))
logger.info('Total time: %i seconds in total (%.1f hours)' % (total_time, total_time_hours))
logger.info('---------------------------------------------------------------------------------------')

logger.info('Missed @ Epoch %i:' % evl.best_test_missed_epoch)
logger.info('  [TEST] F1: %.3f' % evl.best_test_missed)
logger.info('Best @ Epoch %i:' % evl.best_dev_epoch)
logger.info('  [DEV]  F1: %.3f, Recall: %.3f, Precision: %.3f' % (evl.best_dev[0], evl.best_dev[1], evl.best_dev[2]))
logger.info('  [TEST] F1: %.3f, Recall: %.3f, Precision: %.3f' % (evl.best_test[0], evl.best_test[1], evl.best_test[2]))





