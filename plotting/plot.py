import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # for parallel coords
from . import radar_chart as radar_chart_ns

def normalize_colls_0_1(orig_data):
	data = orig_data - np.min(orig_data, axis=0)
	return data / np.max(data, axis=0)

class scatter:
	def __init__(self, points, labels=None, alpha=0.7):
		self.fig, self.ax = plt.subplots(figsize=(5,5))
		self.ax.scatter(*np.transpose(points), c=labels, alpha=alpha)
	def show(self):
		self.fig.show()
	def save(self, filename='scatter.png'):
		self.fig.savefig(filename)

class parallel_coords:
	def __init__(self, data, order, labels=None):
		if labels is None:
			labels = [0 for _ in range(len(data))]
		self.fig, self.ax = plt.subplots(facecolor='white')
		self.ax.axis('off')
		prepped_data = np.transpose(np.vstack([labels, np.transpose(normalize_colls_0_1(data))[order]]))
		self.ax = pd.tools.plotting.parallel_coordinates(pd.DataFrame(prepped_data), 0, axvlines=False)
	def show(self):
		self.fig.show()
	def save(self, filename='parallel_coords.png'):
		self.fig.savefig(filename)

class radar_chart:
	def __init__(self, data, order, labels=None):
		if labels is None:
			labels = [0 for _ in range(len(data))]
		prepped_data = np.transpose(np.transpose(normalize_colls_0_1(data))[order])
		self.fig, self.ax = radar_chart_ns.radar_chart(prepped_data, labels)
	def show(self):
		self.fig.show()
	def save(self, filename='radar_chart.png'):
		self.fig.savefig(filename)
