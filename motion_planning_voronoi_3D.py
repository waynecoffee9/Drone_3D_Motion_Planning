import argparse
import time
import msgpack
import numpy.linalg as LA

from enum import Enum, auto
from shapely.geometry import Polygon, Point, LineString

import numpy as np

from planning_utils import a_star, heuristic, create_grid, random_sampling, chk_node_collide, create_graph
from planning_utils import prune_path, get_heading, load_graph, closest_point, random_goal
from planning_utils import random_graph_pts
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

# Goal in global coordinates - manual input
# GOAL_COORD = (-122.39987, 37.79696, 60)
# (-122.399912, 37.796985)
# (-122.398452, 37.793480)
class States(Enum):
	MANUAL = auto()
	ARMING = auto()
	TAKEOFF = auto()
	WAYPOINT = auto()
	LANDING = auto()
	DISARMING = auto()
	PLANNING = auto()


class MotionPlanning(Drone):

	def __init__(self, connection):
		super().__init__(connection)

		self.target_position = np.array([0.0, 0.0, 0.0])
		self.waypoints = []
		self.in_mission = True
		self.check_state = {}
		self.t0 = None
		self.takeoff_alt = 0
		# initial state
		self.flight_state = States.MANUAL

		# register all your callbacks here
		self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
		self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
		self.register_callback(MsgID.STATE, self.state_callback)

	def local_position_callback(self):
		speed = LA.norm(np.array(self.local_velocity))
		if self.flight_state == States.TAKEOFF:
			if abs(self.local_position[2] + self.takeoff_alt) < 0.3:
				self.waypoint_transition()
		elif self.flight_state == States.WAYPOINT:
			temp_local = self.local_position
			temp_local[2] = -1*temp_local[2]
			if LA.norm(self.target_position[0:3] - temp_local[0:3]) < max(speed*1.2, 1):
				if len(self.waypoints) > 0:
					self.waypoint_transition()
				else:
					if np.linalg.norm(self.local_velocity[0:3]) < 0.5:
						self.landing_transition()

	def velocity_callback(self):
		if self.flight_state == States.LANDING:
			if -1*self.local_position[2] < self.target_position[2]:
				if abs(self.local_velocity[2]) < 0.005:
					# make sure the drone is actually parked by checking again at 1 sec delay
					if not self.t0:
						self.t0 = time.time()
					else:
						if time.time() - self.t0 > 1.0:
							self.disarming_transition()
			#if self.global_position[2] - self.global_home[2] < 0.1:
			#	if abs(self.local_position[2]) < 0.01:

	def state_callback(self):
		if self.in_mission:
			if self.flight_state == States.MANUAL:
				self.arming_transition()
			elif self.flight_state == States.ARMING:
				if self.armed:
					self.plan_path()
			elif self.flight_state == States.PLANNING:
				self.takeoff_transition()
			elif self.flight_state == States.DISARMING:
				if ~self.armed & ~self.guided:
					self.manual_transition()

	def arming_transition(self):
		self.flight_state = States.ARMING
		print("arming transition")
		self.arm()
		self.take_control()

	def takeoff_transition(self):
		self.flight_state = States.TAKEOFF
		print("takeoff transition")
		self.takeoff_alt = self.target_position[2]-self.local_position[2]
		self.takeoff(self.takeoff_alt)

	def waypoint_transition(self):
		self.flight_state = States.WAYPOINT
		print("waypoint transition")
		self.target_position = self.waypoints.pop(0)
		print('target position', self.target_position)
		self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

	def landing_transition(self):
		self.flight_state = States.LANDING
		print("landing transition")
		self.land()

	def disarming_transition(self):
		self.flight_state = States.DISARMING
		print("disarm transition")
		self.disarm()
		self.release_control()

	def manual_transition(self):
		self.flight_state = States.MANUAL
		print("manual transition")
		self.stop()
		self.in_mission = False

	def send_waypoints(self):
		print("Sending waypoints to simulator ...")
		data = msgpack.dumps(self.waypoints)
		self.connection._master.write(data)

	def plan_path(self):
		self.flight_state = States.PLANNING
		print("Searching for a path ...")
		TARGET_ALTITUDE = 10
		SAFETY_DISTANCE = 5

		self.target_position[2] = TARGET_ALTITUDE

		# TODO: read lat0, lon0 from colliders into floating point values
		filename = 'colliders.csv'
		data = np.genfromtxt(filename, delimiter=',', dtype='str', max_rows=1)
		lat0 = float(data[0].split(' ')[-1])
		lon0 = float(data[1].split(' ')[-1])
		# TODO: set home position to (lon0, lat0, 0)
		self.set_home_position(lon0, lat0, 0.0)
		# TODO: retrieve current global position
		# TODO: convert to current local position using global_to_local()
		print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
																		 self.local_position))
		# Read in obstacle map
		data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
		
		# Define a grid for a particular altitude and safety margin around obstacles
		grid, north_offset, east_offset = create_grid(data, 0, SAFETY_DISTANCE)
		print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
		
		# Define starting point on the grid (this is just grid center)
		grid_start = self.global_position
		
		# TODO: convert start position to current position rather than map center
		grid_start = self.global_to_map(grid_start, grid_start, grid.shape, north_offset, east_offset, int(np.ceil(TARGET_ALTITUDE-self.local_position[2])))

		# Set goal as some arbitrary position on the grid
		# generate a random goal
		x_min = 0
		x_max = grid.shape[0]
		y_min = 0
		y_max = grid.shape[1]
		grid_goal = random_goal(grid, x_min, x_max, y_min, y_max)

		# If a goal is input manually, use the global variable GOAL_COORD in (long, lat)
		# grid_goal = GOAL_COORD
		# grid_goal = self.global_to_map(grid_goal, self.global_position, grid.shape, north_offset, east_offset, 60)

		# Run A* to find a path from start to goal
		# TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
		# or move to a different search space such as a graph (not done here)
		print('Local Start and Goal: ', grid_start, grid_goal)
		
		# Use Voronoi graph
		# load a graph
		g = load_graph('map.gpickle')
		print("Number of edges", len(g.edges))
		
		# find closest points for start and goal
		start_closest = closest_point(g, grid_start)
		goal_closest = closest_point(g, grid_goal)
		
		# a-star.  If a path cannot be found, add graph points and paths around start/goal.
		path_found = False
		for i in range(3):
			path, cost, path_found = a_star(g, heuristic, start_closest, goal_closest)
			if path_found:
				break
			g = random_graph_pts(grid, g, grid_start, samples=100)
			g = random_graph_pts(grid, g, grid_goal, samples=100)
			start_closest = closest_point(g, grid_start)
			goal_closest = closest_point(g, grid_goal)
			

		# TODO: prune path to minimize number of waypoints
		pruned_path = prune_path(path, grid)
		print('unpruned_path: ', path, type(path))
		print('pruned_path: ', pruned_path, type(pruned_path))
		
		# append destination to path, if destination is more than 1 meter away from last waypoint
		if LA.norm(np.array(pruned_path[-1]) - np.array(grid_goal)) > 1.0:
			pruned_path.append((grid_goal[0], grid_goal[1], grid_goal[2]))
		
		# assign a heading at each waypoint based on direction of next waypoint
		path_with_heading = get_heading(pruned_path)
		
		# Convert path to waypoints
		waypoints = [[p[0] + north_offset, p[1] + east_offset, p[2], p[3]] for p in path_with_heading]
		
		# Set self.waypoints
		self.waypoints = waypoints
		
		# TODO: send waypoints to sim (this is just for visualization of waypoints)
		self.send_waypoints()

	def global_to_map(self, global_pos, current_pos, shape, n_offset, e_offset, altitude):
		# convert global position to position in map, which includes north and east offset (bottom left corner is [0,0])
		position = global_to_local(global_pos, self.global_home)
		# quick check if the global position falls outside of given map.  
		# If it falls outside, set the position to current drone position
		if position[0] < n_offset or position[0] >= shape[0] or position[1] < e_offset or position[1] >= shape[1]:
			position = global_to_local(current_pos, self.global_home)
			print('Goal coordinate exceeds current map area, goal is set to current drone position...')

		return (int(position[0]) - n_offset, int(position[1]) - e_offset, altitude)
		
	def start(self):
		self.start_log("Logs", "NavLog.txt")

		print("starting connection")
		self.connection.start()

		# Only required if they do threaded
		# while self.in_mission:
		#	pass

		self.stop_log()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--port', type=int, default=5760, help='Port number')
	parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
	args = parser.parse_args()

	conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
	drone = MotionPlanning(conn)
	time.sleep(1)

	drone.start()
