import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree as mst

import collections
def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def symettrize(arr):
	N = len(arr)
	for i in range(N):
		for j in range(N):
			if arr[i,j] != 0:
				arr[j,i] = arr[i,j]
	return arr

def mst_graph(matrix):
	''' Reason for adding 'translate' (it's a dirty fix):
	entries in scipy adjacency matrix are not binary, but refer to the weight of the edge.
	Thus we can't distinguish between scipy's encoding of the lack of an edge (which is 0) and an edge weight of 0.
	To resolve this, we make the edge weights positive by translating all weights by some constant,
	which does not change the MST '''
	translate = np.min(matrix) + 1
	return symettrize(mst(matrix+translate).toarray()).tolist()

def dfs_helper(graph, N, v):
	traversal = []
	for child in filter(lambda i: graph[v][i] != 0, range(N)):
		graph[child][v] = 0 # sever edge in opposite direction
		traversal.append(child)
	if not traversal:
		return [v]
	else:
		return [v] + [dfs_helper(graph, N, nv) for nv in traversal] + [v]

def linearize(graph):
	N = len(graph)
	nested = dfs_helper(graph, N, 0)
	return list(flatten(nested))

def get_layout(matrix):
	return linearize(mst_graph(matrix))
