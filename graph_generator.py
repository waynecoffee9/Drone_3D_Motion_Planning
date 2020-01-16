import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import time
from scipy.spatial import Voronoi
from bresenham import bresenham

def create_grid_and_edges(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    along with Voronoi graph edges given obstacle data and the
    drone's altitude.
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
    # Initialize an empty list for Voronoi points
    points = []
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
            int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
            int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
            int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
            int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = alt+d_alt+safety_distance

            # add center of obstacles to points list
            points.append([int(north - north_min), int(east - east_min)])
            #points.append([(north - north_min), (east - east_min)])
    # TODO: create a voronoi graph based on
    # location of obstacle centres
    graph = Voronoi(points)

    # TODO: check each edge from graph.ridge_vertices for collision
    edges = []
    for v in graph.ridge_vertices:
        p1 = graph.vertices[v[0]]
        p2 = graph.vertices[v[1]]
        
        cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        hit = False

        for c in cells:
            # First check if we're off the map
            if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
                hit = True
                break
            # Next check if we're in collision
            if grid[c[0], c[1]] >= drone_altitude:
                hit = True
                break

        # If the edge does not hit on obstacle
        # add it to the list
        if not hit:
            # array to tuple for future graph creation step)
            p1 = (p1[0], p1[1], drone_altitude)
            p2 = (p2[0], p2[1], drone_altitude)
            edges.append((p1, p2))

    return grid, edges
# load the collider file
filename = 'colliders.csv'
data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
print(data)

drone_altitude = 1
safety_distance = 5

# using Voronoi
VERTICAL_D = 10

# max building height is 212 meters. Divide by 10, round up = 22
MAX_LV = 22
G = nx.Graph()
t0 = time.time()
grid, edges = create_grid_and_edges(data, 1, safety_distance)
# copy and paste 22 graphs on top of each other (z-axis) with 10m interval
# and join vertical nodes
for i in range(MAX_LV):
    for e in edges:
        p1 = (e[0][0], e[0][1], (i+1)*VERTICAL_D)
        p2 = (e[1][0], e[1][1], (i+1)*VERTICAL_D)
        dist = LA.norm(np.array(p2) - np.array(p1))
        G.add_edge(p1, p2, weight=dist)
        if i > 0:
            p1_ = (e[0][0], e[0][1], i*VERTICAL_D)
            p2_ = (e[1][0], e[1][1], i*VERTICAL_D)
            G.add_edge(p1, p1_, weight=VERTICAL_D)
            G.add_edge(p2, p2_, weight=VERTICAL_D)
        if i < MAX_LV-1:
            p1_ = (e[0][0], e[0][1], (i+2)*VERTICAL_D)
            p2_ = (e[1][0], e[1][1], (i+2)*VERTICAL_D)
            G.add_edge(p1, p1_, weight=VERTICAL_D)
            G.add_edge(p2, p2_, weight=VERTICAL_D)
print(len(G.edges))            
print('Voronoi took {0} seconds to build'.format(time.time()-t0))

#save graph
nx.write_gpickle(G, "map.gpickle")