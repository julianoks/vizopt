import numpy as np

def correlation_cost(self, dim1, dim2):
	''' cost decreases with correlation, so as to maximize correlation '''
	denom = (np.linalg.norm(dim1) * np.linalg.norm(dim2))
	if denom == 0.0: return 1
	return -np.dot(dim1, dim2) / denom
def euclidean_cost(self, dim1, dim2):
	return np.linalg.norm(dim1 - dim2)
def hard_nearest_neighbor_innacuracy(self, dim1, dim2):
	if self.labels is None:
		import warnings
		warnings.warn("Attempted to use 'hard_nearest_neighbor_innacuracy' without labels.")
		return 42
	n_records = len(dim1)
	centroids = {}
	for i,label in enumerate(self.labels):
		if label not in centroids: centroids[label] = [i]
		else: centroids[label].append(i)
	for key in centroids.keys():
		centroids[key] = np.array([np.mean(dim1[centroids[key]]), np.mean(dim2[centroids[key]])])
	incorrect = 0
	for label, x, y in zip(self.labels, dim1, dim2):
		min_dist, min_label = np.inf, None
		for key in centroids.keys():
			dist = np.linalg.norm(centroids[key] - [x,y])
			if dist < min_dist:
				min_dist, min_label = dist, key
		if min_label != label:
			incorrect += 1
	return incorrect / n_records
def soft_nearest_neighbor_innacuracy(self, dim1, dim2):
	n_records = len(dim1)
	centroids = {}
	for i,label in enumerate(self.labels):
		if label not in centroids: centroids[label] = [i]
		else: centroids[label].append(i)
	for key in centroids.keys():
		centroids[key] = np.array([np.mean(dim1[centroids[key]]), np.mean(dim2[centroids[key]])])
	cost = 0
	for label, x, y in zip(self.labels, dim1, dim2):
		distances = {}
		for key in centroids.keys():
			distances[key] = np.linalg.norm(centroids[key] - [x,y])
		normalizer = np.sum(np.exp(x))
		for key in distances:
			p = np.exp(distances[key]) / normalizer
			if key == label: cost += p - 0
			else: cost += 1 - p
	return cost / n_records

pair_cost_fns = [correlation_cost,euclidean_cost,hard_nearest_neighbor_innacuracy,soft_nearest_neighbor_innacuracy]
