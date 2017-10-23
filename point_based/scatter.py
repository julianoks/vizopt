import tensorflow as tf

def normalize_cols_0_1(orig_data):
	''' Normalizes the data such that, for every column, min(column) = 0, max(column) = 1. '''
	data = orig_data - tf.reduce_min(orig_data, 0)
	remove_zeros = lambda x: tf.cond(x < 1e-5, lambda: 1.0, lambda: x)
	maxs = tf.map_fn(remove_zeros, tf.reduce_max(data, 0))
	return data / maxs

def translate_cols_mean(data):
	return data - tf.reduce_mean(data, 0)


class identity:
	def __init__(self, data, d=2):
		N, M = data.get_shape().as_list()
		self.param_shape = [N,d]
	def view(self, parameters):
		return parameters

class radviz:
	def __init__(self, data, d=None):
		N, M = data.get_shape().as_list()
		self.param_shape = [M]
		self.data = normalize_cols_0_1(data)
		remove_zeros = lambda x: tf.cond(x < 1e-5, lambda: 1.0, lambda: x)
		row_sums = tf.map_fn(remove_zeros, tf.reduce_sum(self.data, 1))
		self.data = tf.transpose(tf.transpose(self.data) / row_sums)
	def view(self, parameters):
		projection = tf.transpose([tf.cos(parameters),  tf.sin(parameters)])
		return tf.matmul(self.data, projection)

class linear_projection:
	def __init__(self, data, d=2):
		N, M = data.get_shape().as_list()
		self.param_shape = [M,d]
		self.data = translate_cols_mean(data)
	def view(self, parameters):
		return tf.matmul(self.data, parameters)
