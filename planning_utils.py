from enum import Enum
from queue import PriorityQueue
import numpy as np
import networkx as nx
import numpy.linalg as LA
import time
from sklearn.neighbors import KDTree

import sys

def create_grid(data, drone_altitude, safety_distance):
	"""
	Returns a grid representation of a 2D configuration space
	based on given obstacle data, drone altitude and safety distance
	arguments.
	"""

	# minimum and maximum north coordinates
	north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
	north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

	# minimum and maximum east coordinates
	east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
	east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

	# given the minimum and maximum coordinates we can
	# calculate the size of the grid.
	north_size = int(np.ceil(north_max - north_min))
	east_size = int(np.ceil(east_max - east_min))

	# Initialize an empty grid
	grid = np.zeros((north_size, east_size))

	# Populate the grid with obstacles
	for i in range(data.shape[0]):
		north, east, alt, d_north, d_east, d_alt = data[i, :]
		obstacle = [
			int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
			int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
			int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
			int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
		]
		grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = alt+d_alt+safety_distance
	grid = grid.astype(int)

	return grid, int(north_min), int(east_min)
	
# Assume all actions cost the same.
class Action(Enum):
	"""
	An action is represented by a 3 element tuple.

	The first 2 values are the delta of the action relative
	to the current grid position. The third and final value
	is the cost of performing the action.
	"""

	WEST = (0, -1, 1)
	EAST = (0, 1, 1)
	NORTH = (-1, 0, 1)
	SOUTH = (1, 0, 1)

	@property
	def cost(self):
		return self.value[2]

	@property
	def delta(self):
		return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
	"""
	Returns a list of valid actions given a grid and current node.
	"""
	valid_actions = list(Action)
	n, m = grid.shape[0] - 1, grid.shape[1] - 1
	x, y = current_node

	# check if the node is off the grid or
	# it's an obstacle

	if x - 1 < 0 or grid[x - 1, y] == 1:
		valid_actions.remove(Action.NORTH)
	if x + 1 > n or grid[x + 1, y] == 1:
		valid_actions.remove(Action.SOUTH)
	if y - 1 < 0 or grid[x, y - 1] == 1:
		valid_actions.remove(Action.WEST)
	if y + 1 > m or grid[x, y + 1] == 1:
		valid_actions.remove(Action.EAST)

	return valid_actions

def a_star(graph, h, start, goal):
	t0 = time.time()
	path = []
	path_cost = 0
	queue = PriorityQueue()
	queue.put((0, start))
	visited = set(start)

	branch = {}
	found = False
	
	while not queue.empty():
		item = queue.get()
		current_node = item[1]
		if current_node == start:
			current_cost = 0.0
		else:			  
			current_cost = branch[current_node][0]
			
		if current_node == goal:		
			print('Found a path.')
			found = True
			break
		else:
			for next_node in graph[current_node]:
				cost = graph.edges[current_node, next_node]['weight']
				branch_cost = current_cost + cost
				queue_cost = branch_cost + h(next_node, goal)
				
				if next_node not in visited:
					visited.add(next_node)
					branch[next_node] = (branch_cost, current_node)
					queue.put((queue_cost, next_node))
			 
	if found:
		# retrace steps
		n = goal
		path_cost = branch[n][0]
		path.append(goal)
		while branch[n][1] != start:
			path.append(branch[n][1])
			n = branch[n][1]
		path.append(branch[n][1])
	else:
		print('**********************')
		print('Failed to find a path!')
		print('**********************') 
	path = [(int(node[0]),int(node[1]),int(node[2])) for node in path]
	print('a-star took {0} seconds to search'.format(time.time()-t0))
	return path[::-1], path_cost, found


def heuristic(position, goal_position):
	return LA.norm(np.array(position) - np.array(goal_position))

def random_sampling(data, start, goal, num_samples=100, altitude=1):
	xmin = int(np.floor(np.min(data[:, 0] - data[:, 3])))
	xmax = int(np.ceil(np.max(data[:, 0] + data[:, 3])))

	ymin = int(np.floor(np.min(data[:, 1] - data[:, 4])))
	ymax = int(np.ceil(np.max(data[:, 1] + data[:, 4])))

	zmin = altitude
	# Limit the z axis for the visualization
	zmax = altitude + 1

	xvals = np.random.randint(0, xmax-xmin, num_samples)
	yvals = np.random.randint(0, ymax-ymin, num_samples)
	zvals = np.random.randint(zmin, zmax, num_samples)

	samples = list(zip(xvals, yvals, zvals))
	start_ne_int = (start[0], start[1], zmin)
	goal_ne_int = (goal[0], goal[1], zmin)
	samples.append(start_ne_int)
	samples.append(goal_ne_int)
	return samples

def collides(grid, point):
	if grid[point[0], point[1]] > point[2]:
		return True
	else: return False

def chk_node_collide(samples, grid):
	t0 = time.time()
	to_keep = []
	for point in samples:
		if not collides(grid, point):
			to_keep.append(point)
	print('Checking for node colliding took {0} seconds to finish'.format(time.time()-t0))
	return to_keep

def can_connect(grid, p1, p2):
	cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
	for cell in cells:
		if grid[cell[0],cell[1]] > min(p1[2], p2[2]):
			return False
	return True

def create_graph(grid, to_keep, k):
	t0 = time.time()
	g = nx.Graph()
	tree = KDTree(np.array(to_keep))
	for n1 in to_keep:
		dist, ids = tree.query([n1], k)
		dist = dist[0][0::3]
		ids = ids[0][0::3]
		dist_rev = list(reversed(dist))
		ids_rev = list(reversed(ids))
		count = 0
		for i in range(len(ids)):
			if count > 5:
				break
			n2 = to_keep[ids[i]]
			if n1 == n2:
				continue
			if can_connect(grid, n1, n2):
				g.add_edge(n1, n2, weight=dist[i])
				count += 1
			n2 = to_keep[ids_rev[i]]
			if n1 == n2:
				continue
			if can_connect(grid, n1, n2):
				g.add_edge(n1, n2, weight=dist_rev[i])
				count += 1
	print('graph took {0} seconds to build'.format(time.time()-t0))
	return g

def load_graph(filename):
	return nx.read_gpickle(filename)

def prune_path(path, grid):
	if path is not None:
		pruned_path = [p for p in path]
		# TODO: prune the path!
		p1_i = 0
		p2_i = 1
		while p2_i < (len(pruned_path)):
			p1 = pruned_path[p1_i]
			p2 = pruned_path[p2_i]
			cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
			for cell in cells:
				if grid[cell[0],cell[1]] > min(p1[2], p2[2]):
					#print(cell[0], cell[1])
					del pruned_path[p1_i+1:p2_i-1]
					p1_i += 1
					p2_i = p1_i
					break
			p2_i += 1
			if p2_i == len(pruned_path) and p2_i - p1_i > 1:
				del pruned_path[p1_i+1:p2_i-1]
				
	else:
		pruned_path = path
		
	return pruned_path
	
def get_heading(path):
	count = 1
	path[0] = path[0] + (0.0,)
	path_pairs = zip(path[:-1], path[1:])
	for (n1, n2) in path_pairs:
		heading = np.arctan2(n2[1]-n1[1], n2[0]-n1[0])
		path[count] = path[count] + (heading,)
		count += 1
	#path[-1] = path[-1] + (0,)
	return path
	
def closest_point(graph, current_node):
	tree = KDTree(np.array(list(graph.nodes)))
	dist, ids = tree.query([current_node], 1)
	ids = ids[0]
	
	return list(graph.nodes)[ids[0]]

def bresenham(x0, y0, x1, y1):
	
	"""Yield integer coordinates on the line from (x0, y0) to (x1, y1).
	Input coordinates should be integers.
	The result will contain both the start and the end point.
	
	This is a customized version which is more conservative (yields more coordinates)
	"""
	dx = x1 - x0
	dy = y1 - y0

	xsign = 1 if dx > 0 else -1
	ysign = 1 if dy > 0 else -1

	dx = abs(dx)
	dy = abs(dy)

	if dx > dy:
		xx, xy, yx, yy = xsign, 0, 0, ysign
	else:
		dx, dy = dy, dx
		xx, xy, yx, yy = 0, ysign, xsign, 0

	D = 2*dy - dx
	if dy == 0 and dx == 0:
		return
	m = dy/dx
	y = 0
	
	for x in range(dx + 1):
		yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
		if m*(x+1) > y+1:
			yield x0 + x*xx + (y+1)*yx, y0 + x*xy + (y+1)*yy
			y += 1
		elif m*(x+1) == y+1 and x < dx:
			yield x0 + x*xx + (y+1)*yx, y0 + x*xy + (y+1)*yy
			yield x0 + (x+1)*xx + y*yx, y0 + (x+1)*xy + y*yy
			y += 1

def random_goal(grid, x_min, x_max, y_min, y_max):
	x = np.random.randint(x_min, x_max)
	y = np.random.randint(y_min, y_max)
	z_min = grid[x, y] + 1
	z_max = z_min + 5
	z = np.random.randint(z_min, z_max)
	return (x, y, z)
	
def random_graph_pts(grid, G, goal, samples=500):
	t0 = time.time()
	search_dis = 150
	search_sq = [
	int(np.clip(goal[0] - search_dis, 0, grid.shape[0]-1)),
	int(np.clip(goal[0] + search_dis, 0, grid.shape[0]-1)),
	int(np.clip(goal[1] - search_dis, 0, grid.shape[1]-1)),
	int(np.clip(goal[1] + search_dis, 0, grid.shape[1]-1)),
	]
	
	max_height = int(max(grid[search_sq[0]:search_sq[1], search_sq[2]:search_sq[3]].flatten()))
	max_height = max(max_height, int(goal[2]))
	num_samples = samples
	xvals = np.random.randint(search_sq[0], search_sq[1], num_samples)
	yvals = np.random.randint(search_sq[2], search_sq[3], num_samples)
	zvals = np.random.randint(0, max_height+10, num_samples)
	samples = list(zip(xvals, yvals, zvals))
	
	to_keep = []
	for point in samples:
		if grid[point[0], point[1]] <= point[2]:
			to_keep.append(point)
	
	G = add_graph(G, grid, to_keep, 25)

	return G
	
def add_graph(G, grid, to_keep, k):
	tree2 = KDTree(np.array(G.nodes))
	for n1 in to_keep:
		dist, ids = tree2.query([n1], k)
		dist = dist[0][0::3]
		ids = ids[0][0::3]
		dist_rev = list(reversed(dist))
		ids_rev = list(reversed(ids))
		count = 0
		for i in range(len(ids)):
			if count > 10:
				break
			n2 = list(G.nodes)[ids[i]]
			if n1 == n2:
				continue
			if can_connect(grid, n1, n2):
				G.add_edge(n1, n2, weight=dist[i])
				count += 1
			n2 = list(G.nodes)[ids_rev[i]]
			if n1 == n2:
				continue
			if can_connect(grid, n1, n2):
				G.add_edge(n1, n2, weight=dist_rev[i])
				count += 1
	return G