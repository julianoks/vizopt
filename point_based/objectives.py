import tensorflow as tf

def dist_matrix(data):
	r = tf.reshape(tf.reduce_sum(data * data, 1), [-1,1])
	d = r - 2*tf.matmul(data, tf.transpose(data)) + tf.transpose(r)
	return tf.matrix_set_diag(d, tf.ones(d.shape[0]))

def frobenius_norm(matrix):
    return tf.reduce_sum(matrix ** 2) ** 0.5

class sammon_stress:
	def __init__(self, data):
		N, M = data.get_shape().as_list()
		self.original_dist = dist_matrix(data)
		self.denom = tf.reduce_sum(self.original_dist)
	def cost(self, scatterplot):
		''' negated so as to maximize '''
		emb_dist = dist_matrix(scatterplot)
		return tf.reduce_sum(frobenius_norm(emb_dist - self.original_dist) / self.original_dist) / self.denom

class collapse:
	def __init__(self, data, labels):
		N, M = data.get_shape().as_list()
		self.same_class = tf.stack([[1.0 if i==j else 0.0 for j in labels]for i in labels])
		self.different_class = tf.stack([[0.0 if i==j else 1.0 for j in labels]for i in labels])
	def cost(self, scatterplot):
		''' negated so as to maximize '''
		dist = dist_matrix(scatterplot)
		lambd = 0.5
		cost = (1 - lambd) * tf.reduce_sum(self.same_class * dist)
		cost -= lambd * tf.reduce_sum(self.different_class * dist)
		return cost

class centroid:
	def __init__(self): None
	def cost(self, scatterplot):
		mean = tf.reduce_mean(scatterplot, 0)
		distances = tf.norm(scatterplot - mean, axis=1)
		return - tf.reduce_sum(distances)
