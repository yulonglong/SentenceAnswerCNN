import keras.backend as K
from keras.engine.topology import Layer
from keras.layers.convolutional import Convolution1D

class MeanOverTime(Layer):
	def __init__(self, mask_zero=True, **kwargs):
		self.mask_zero = mask_zero
		self.supports_masking = True
		super(MeanOverTime, self).__init__(**kwargs)

	def call(self, x, mask=None):
		if self.mask_zero:
			return K.cast((x.sum(axis=1) / (x.shape[1] - K.equal(x, 0).all(axis=2).sum(axis=1, keepdims=True))), K.floatx())
		else:
			return K.mean(x, axis=1)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[2])
	
	def compute_mask(self, x, mask):
		return None
	
	def get_config(self):
		config = {'mask_zero': self.mask_zero}
		base_config = super(MeanOverTime, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class MulConstant(Layer):
	def __init__(self, coef=1, **kwargs):
		self.coef = coef
		super(MulConstant, self).__init__(**kwargs)

	def call(self, x, mask=None):
		return self.coef * x

	def get_output_shape_for(self, input_shape):
		return input_shape
	
	def get_config(self):
		config = {'coef': self.coef}
		base_config = super(MulConstant, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class Conv1DWithMasking(Convolution1D):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(Conv1DWithMasking, self).__init__(**kwargs)

	#def call(self, x, mask=None):
	#	if self.mask_zero:
	#		return K.cast((x.sum(axis=1) / (x.shape[1] - K.equal(x, 0).all(axis=2).sum(axis=1, keepdims=True))), K.floatx())
	#	else:
	#		return K.mean(x, axis=1)

	#def get_output_shape_for(self, input_shape):
	#	return (input_shape[0], input_shape[2])
	
	def compute_mask(self, x, mask):
		return mask
	
	#def get_config(self):
	#	#config = {'mask_zero': self.mask_zero}
	#	base_config = super(Conv1DWithMasking, self).get_config()
	#	return dict(list(base_config.items()) + list(config.items()))

################################################################################################################################################
## Depricated functions
#

from keras.layers.core import Lambda

def MeanOverTime_depricated(mask_zero=True):
	if mask_zero:
		# Masks the timestep vector if all elements are zero
		mean_func = lambda x: K.cast((x.sum(axis=1) / (x.shape[1] - K.equal(x, 0).all(axis=2).sum(axis=1, keepdims=True))), K.floatx())
		layer = Lambda(mean_func, output_shape=lambda s: (s[0], s[2]))
	else:
		# Even if the timestep vector is all zeros, it will be used in averaging (so a notion of sequence length is preserved)
		layer = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[2]))
	layer.supports_masking = True
	#layer.name = 'MeanOverTime'
	def compute_mask(input, mask):
		return None
	layer.compute_mask = compute_mask
	return layer

def MulConstant_depricated(coef):
	layer = Lambda(lambda x: coef * x, output_shape=lambda s: s)
	#layer.name = 'MulConstant'
	return layer
