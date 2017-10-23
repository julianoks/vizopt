import tensorflow as tf
import numpy as np
from . import scatter
from . import objectives
from plotting import plot

def optimize(data, visualization, objective, n_iterations=1000, learning_rate=1):
	theta = tf.Variable(tf.random_uniform(visualization.param_shape), name="theta")
	view = visualization.view(theta)
	cost = objective.cost(view)
	with tf.Session() as sess:
		opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
		minimizer = opt.minimize(cost, var_list=[theta])
		sess.run(tf.global_variables_initializer())
		[minimizer.run() for _ in range(n_iterations)]
		return np.array(view.eval())

def save_tabular_report(data, labels=None, pathname=''):
	tf_data = tf.convert_to_tensor(data, dtype=tf.float32) #tf.stack(data.astype(np.float32))
	point_based_modules = {'visualization':
								{'embedding': scatter.identity(tf_data),
								'projection': scatter.linear_projection(tf_data),
								'radviz': scatter.radviz(tf_data)},
							'objectives':
								{'sammon': objectives.sammon_stress(tf_data),
								'centroid': objectives.centroid()}}
	if labels is not None:
		point_based_modules['objectives']['collapse'] = objectives.collapse(tf_data, labels)
	for viz_name, viz_func in point_based_modules['visualization'].items():
		for obj_name, obj_func in point_based_modules['objectives'].items():
			print(viz_name,obj_name)
			view = optimize(tf_data, viz_func, obj_func)
			plot.scatter(view, labels=labels).save(filename = pathname+'/'+viz_name+'_'+obj_name+'.png')
