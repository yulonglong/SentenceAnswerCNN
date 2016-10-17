import asap_reader as asap
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
###############################################################################################################################
## Evaluator class
#

class Evaluator():
	
	def __init__(self, logger, out_dir, train, dev, test, model_type, batch_size_eval = 256, print_info=False):
		self.logger = logger
		self.out_dir = out_dir
		self.model_type = model_type
		self.batch_size_eval = batch_size_eval

		self.train_x, self.train_y = train[0], train[1]
		self.dev_x, self.dev_y = dev[0], dev[1]
		self.test_x, self.test_y = test[0], test[1]
		self.dev_mean = self.dev_y.mean()
		self.dev_std = self.dev_y.std()
		self.test_mean = self.test_y.mean()
		self.test_std = self.test_y.std()

		self.train_y_org = self.train_y.astype('int32')
		self.dev_y_org = self.dev_y.astype('int32')
		self.test_y_org = self.test_y.astype('int32')

		self.best_dev = [-1, -1, -1, -1]
		self.best_test = [-1, -1, -1, -1]
		self.best_dev_epoch = -1
		self.best_test_missed = -1
		self.best_test_missed_epoch = -1
		self.dump_ref_scores()
	
	def dump_ref_scores(self):
		np.savetxt(self.out_dir + '/preds/train_ref.txt', self.train_y_org, fmt='%i')
		np.savetxt(self.out_dir + '/preds/dev_ref.txt', self.dev_y_org, fmt='%i')
		np.savetxt(self.out_dir + '/preds/test_ref.txt', self.test_y_org, fmt='%i')
	
	def dump_predictions(self, train_pred, dev_pred, test_pred, epoch):
		np.savetxt(self.out_dir + '/preds/train_pred_' + str(epoch) + '.txt', train_pred, fmt='%.8f')
		np.savetxt(self.out_dir + '/preds/dev_pred_' + str(epoch) + '.txt', dev_pred, fmt='%.8f')
		np.savetxt(self.out_dir + '/preds/test_pred_' + str(epoch) + '.txt', test_pred, fmt='%.8f')
	
	def evaluate(self, model, epoch):
		self.dev_loss, self.dev_metric = model.evaluate(self.dev_x, self.dev_y, batch_size=self.batch_size_eval, verbose=0)
		self.test_loss, self.test_metric = model.evaluate(self.test_x, self.test_y, batch_size=self.batch_size_eval, verbose=0)
		
		self.train_pred = model.predict(self.train_x, batch_size=self.batch_size_eval).squeeze()
		self.dev_pred = model.predict(self.dev_x, batch_size=self.batch_size_eval).squeeze()
		self.test_pred = model.predict(self.test_x, batch_size=self.batch_size_eval).squeeze()

		self.dump_predictions(self.train_pred, self.dev_pred, self.test_pred, epoch)

		# If it is a binary classification
		binary_train_pred = self.train_pred
		high_indices = binary_train_pred >= 0.5
		binary_train_pred[high_indices] = 1
		low_indices = binary_train_pred < 0.5
		binary_train_pred[low_indices] = 0

		self.train_recall = recall_score(self.train_y_org,binary_train_pred)
		self.train_precision = precision_score(self.train_y_org,binary_train_pred)
		self.train_f1= f1_score(self.train_y_org,binary_train_pred)
		
		# If it is a binary classification
		binary_dev_pred = self.dev_pred
		high_indices = binary_dev_pred >= 0.5
		binary_dev_pred[high_indices] = 1
		low_indices = binary_dev_pred < 0.5
		binary_dev_pred[low_indices] = 0

		self.dev_recall = recall_score(self.dev_y_org,binary_dev_pred)
		self.dev_precision = precision_score(self.dev_y_org,binary_dev_pred)
		self.dev_f1= f1_score(self.dev_y_org,binary_dev_pred)

		binary_test_pred = self.test_pred
		high_indices = binary_test_pred >= 0.5
		binary_test_pred[high_indices] = 1
		low_indices = binary_test_pred < 0.5
		binary_test_pred[low_indices] = 0

		self.test_recall = recall_score(self.test_y_org,binary_test_pred)
		self.test_precision = precision_score(self.test_y_org,binary_test_pred)
		self.test_f1= f1_score(self.test_y_org,binary_test_pred)

		if (self.dev_f1 > self.best_dev[0]):
			self.best_dev = [self.dev_f1, self.dev_recall, self.dev_precision]
			self.best_test = [self.test_f1, self.test_recall, self.test_precision]
			self.best_dev_epoch = epoch
			model.save_weights(self.out_dir + '/best_model_weights.h5', overwrite=True)

		if (self.test_f1 > self.best_test_missed):
			self.best_test_missed = self.test_f1
			self.best_test_missed_epoch = epoch

	def print_info(self):
		self.logger.info('[Dev]   loss: %.4f, metric: %.4f, mean: %.3f (%.3f), stdev: %.3f (%.3f)' % (
			self.dev_loss, self.dev_metric, self.dev_pred.mean(), self.dev_mean, self.dev_pred.std(), self.dev_std))
		self.logger.info('[Test]  loss: %.4f, metric: %.4f, mean: %.3f (%.3f), stdev: %.3f (%.3f)' % (
			self.test_loss, self.test_metric, self.test_pred.mean(), self.test_mean, self.test_pred.std(), self.test_std))
		self.logger.info('[TRAIN] F1: %.3f, Recall: %.3f, Precision: %.3f' % (
			self.train_f1, self.train_recall, self.train_precision))
		self.logger.info('[DEV]   F1: %.3f, Recall: %.3f, Precision: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f)' % (
			self.dev_f1, self.dev_recall, self.dev_precision, self.best_dev_epoch,
			self.best_dev[0], self.best_dev[1], self.best_dev[2]))
		self.logger.info('[TEST]  F1: %.3f, Recall: %.3f, Precision: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f)' % (
			self.test_f1, self.test_recall, self.test_precision, self.best_dev_epoch,
			self.best_test[0], self.best_test[1], self.best_test[2]))
		self.logger.info('---------------------------------------------------------------------------------------')

		content = ("\r\n\r\n[TRAIN] F1: %.3f, Recall: %.3f, Precision: %.3f" % (
			self.train_f1, self.train_recall, self.train_precision))
		content = content + ("\r\n\r\n[DEV]   F1: %.3f, Recall: %.3f, Precision: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f)" % (
			self.dev_f1, self.dev_recall, self.dev_precision, self.best_dev_epoch,
			self.best_dev[0], self.best_dev[1], self.best_dev[2]))
		content = content + ("\r\n\r\n[TEST]  F1: %.3f, Recall: %.3f, Precision: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f)" % (
			self.test_f1, self.test_recall, self.test_precision, self.best_dev_epoch,
			self.best_test[0], self.best_test[1], self.best_test[2]))
		return content
