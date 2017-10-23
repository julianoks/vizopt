import numpy as np
import bisect, random
from . import mst, pair_cost

pair_cost_fns = pair_cost.pair_cost_fns

def normalize_colls_0_1(data):
	''' Normalizes the data such that, for any column, min(column) = 0, max(column) = 1. '''
	mins = np.min(data, axis=0)
	maxs = np.max(data, axis=0)
	return (data - mins) / [1 if x==0 else x for x in (maxs - mins)]

class ordering:
	def __init__(self, data, labels=None):
		self.N, self.M = np.shape(data)
		self.data = normalize_colls_0_1(data)
		self.labels = labels
		self.pair_cost = pair_cost.euclidean_cost
	def make_cost_matrix(self):
		''' assumes cost matrix is symmetric, and leaves diagonal to inf '''
		mat = np.zeros([self.M, self.M])
		for i in range(self.M):
			mat[i,i] = np.inf
			for j in range(i):
				cost = self.pair_cost(self, self.data[:,i], self.data[:,j])
				mat[i,j] = cost
				mat[j,i] = cost
		self.cost_matrix = mat
		return
	def optimize_mst(self):
		return mst.get_layout(self.cost_matrix)
	def optimize_astar_tsp(self):
		self.make_cost_matrix()
		all_indices = set(range(self.M))
		def heuristic(solution_list):
			remaining = list(all_indices.difference(solution_list[:-1]))
			return np.sum(np.min(self.cost_matrix[remaining].T[remaining], axis=0))
		def successors(solution_obj):
			last_i = solution_obj[0][-1]
			for i in all_indices.difference(solution_obj[0]):
				new_solution = solution_obj[0] + [i]
				g = solution_obj[1] + self.cost_matrix[last_i, i]
				h = heuristic(new_solution)
				f = g + h
				new_obj = [new_solution, g, h, f]
				yield new_obj
		queue = [[[i],0,0,0] for i in all_indices] # each element is [solution, g, h, f]
		queue_scores = [0 for _ in all_indices]
		iteration = 0
		while True:
			iteration += 1
			best = queue.pop(0)
			queue_scores.pop(0) #print(iteration, len(best[0]), queue_scores.pop(0))
			if len(best[0]) == self.M:
				print("Done!  Path cost:", best[1])
				return best[0]
			for child in successors(best):
				index = bisect.bisect(queue_scores, child[3])
				queue_scores = queue_scores[:index] + [child[3]] + queue_scores[index:]
				queue = queue[:index] + [child] + queue[index:]
				queue_scores = queue_scores[:1000]
				queue = queue[:1000]
	def optimize_hill_climb_tsp(self):
		self.make_cost_matrix()
		def get_score(perm):
			return sum([self.cost_matrix[perm[i],perm[i+1]] for i in range(self.M-1)])
		solution = list(range(self.M))
		current_score = get_score(solution)
		for i in range(500000):
			p1, p2 = sorted(random.sample(range(self.M), 2))
			delta = 0
			delta += self.cost_matrix[solution[p2], solution[p1+1]] - self.cost_matrix[solution[p1], solution[p1+1]]
			if p1 != 0:
				delta += self.cost_matrix[solution[p1-1], solution[p2]] - self.cost_matrix[solution[p1-1], solution[p1]]
			delta += self.cost_matrix[solution[p2-1], solution[p1]] - self.cost_matrix[solution[p2-1], solution[p2]]
			if p2 != self.M-1:
				delta += self.cost_matrix[solution[p1], solution[p2+1]] - self.cost_matrix[solution[p2], solution[p2+1]]
			if delta <= 0:
				hold = solution[p2]
				solution[p2] = solution[p1]
				solution[p1] = hold
		return solution
