from plotting import plot
import line_based.ordering
import point_based.scatter
import point_based.objectives
import point_based.optimize

if __name__ == '__main__':
	pathname='images'
	# generate data
	import sklearn.datasets
	from sklearn.utils import shuffle as collated_shuffle
	n_records = 100
	data, labels = sklearn.datasets.make_blobs(n_records, n_features=100, cluster_std=0.1, centers=3, center_box=(0,1))
	#data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)
	data, labels = collated_shuffle(data, labels)
	data, labels = data[:n_records], labels[:n_records]
	print("Data loaded...")
	# line based
	for cost_fn in line_based.ordering.pair_cost_fns:
		line_based_obj = line_based.ordering.ordering(data, labels=labels)
		line_based_obj.pair_cost = cost_fn
		line_based_obj.make_cost_matrix()
		optimized_ordering = line_based_obj.optimize_mst()
		print("Optimized ordering for", cost_fn.__name__)
		plot.parallel_coords(data, optimized_ordering, labels=labels).save(filename=pathname+'/parallel_coords_'+cost_fn.__name__+'.png')
		plot.radar_chart(data, optimized_ordering[:-1], labels=labels).save(filename=pathname+'/radar_chart_'+cost_fn.__name__+'.png')
	# point based
	point_based.optimize.save_tabular_report(data, labels=labels, pathname=pathname)
