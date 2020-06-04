from gym import spaces
from math import cos, sin, pi, tan, atan
import numpy as np
import random
from PIL import Image
import cv2
from shapely.geometry import Polygon, Point, LinearRing, LineString


def rotate_points(point_in, angle, centre):
	ox, oy = centre
	point_out = []
	for point in point_in:
		px, py = point
		cs = cos(angle)
		sn = sin(angle)
		xd = px - ox
		yd = py - oy
		qx = ox + cs * xd - sn * yd
		qy = oy + sn * xd + cs * yd
		point_out.append((qx, qy))

	return point_out

def wrap_points(list_in):
	out=list_in
	out.append(out[0])
	return out

class CarGameEnv:

	def __init__(self, continuous_flag):
		# initialise track
		# track_coords = [(100, 200), (100, 500), (500, 500), (500, 100), (100, 100)]
		# track_coords = [(100, 200), (100, 500), (300, 500), (300+141, 300+141), (300+200, 300), (300+141, 300-141), (300, 100), (100, 100)]
		track_coords = [(100, 200), (100, 500), (500, 500), (500, 400), (400, 250), (500, 100), (100, 100)]
		track_width = 40
		self.middle_poly = Polygon(track_coords)
		self.middle_lin = LinearRing(self.middle_poly.exterior.coords)
		temp_coords = wrap_points([(x, y) for x,y in zip(*self.middle_poly.exterior.xy)])
		self.middle_coords = temp_coords[::-1]
		temp_poly = Polygon(self.middle_poly.buffer(track_width*-1).exterior, [self.middle_lin])
		temp_coords = wrap_points([(x, y) for x,y in zip(*temp_poly.exterior.xy)])
		self.inner_coords = temp_coords[::-1]
		self.inner_poly = Polygon(self.inner_coords)
		self.inner_lin = LinearRing(self.inner_poly.exterior.coords)
		temp_poly = Polygon(self.middle_poly.buffer(track_width).exterior, [self.middle_lin])
		temp_coords = wrap_points([(x, y) for x,y in zip(*temp_poly.exterior.xy)])
		self.outer_coords = temp_coords[::-1]
		self.outer_poly = Polygon(self.outer_coords)
		self.outer_lin = LinearRing(self.outer_poly.exterior.coords)
		self.lap_length = self.middle_poly.length
		print(self.lap_length)
		# initialise car
		self.sense_ang = np.array([0])
		self.sense_ang = np.linspace(-90, 90, num=9, endpoint = True)
		self.mass = 750
		self.aero_drag_v2 = 0.5*1.225*1.3
		self.aero_down_v2 = self.aero_drag_v2*2.5
		self.forward_force = 8000
		self.vel_max = (self.forward_force/self.aero_drag_v2)**0.5
		self.wheelbase = 3.7
		self.steer_lock_ang = atan(self.wheelbase/30)
		self.friction_coeff = 1.6
		self.time_per_frame = 0.2
		self.sz = (16, 32)
		# misc
		self.viewer = None
		self.win_sz = (608,608)
		self.win_diag = (self.win_sz[0]**2+self.win_sz[1]**2)**0.5
		self.patience = 10
		self.lap_targ = 2
		self.loc_mem_sz = 50
		self.dist_mem_ind = list(range(0,10,2))
		# self.dist_mem_ind = [0]
		# reset
		self.reset()
		# spaces
		high = np.ones(len(self.state))
		self.observation_space = spaces.Box(-high, high, dtype=np.float32)
		self.continuous = continuous_flag
		if self.continuous:
			if self.sense_ang.size > 1:
				self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
			else:
				self.action_space = spaces.Box(-1, +1, (1,), dtype=np.float32)
		else:
			if self.sense_ang.size > 1:
				self.actions_avail = [(1,0) , (-1,0), (0,0), (0,-1), (0,1)]
			else:
				self.actions_avail = [(1,) , (-1,), (0,)]
			self.action_space = spaces.Discrete(len(self.actions_avail))

	def reset(self):
		self.finished_episode = False
		self.frame_curr = 0
		self.bear = 0
		self.vel = 5
		self.lap_whole = 0
		self.lap_float = None
		self.lap_float_max = self.lap_float
		self.lap_float_frame_max = self.frame_curr
		self.track_prog_prev = None
		self.score = None
		self.hitbox_center = self.middle_coords[0]
		self.hitbox_coords = self.outer_points()
		self.loc_mem = [self.hitbox_center]*self.loc_mem_sz
		self.pos_analyse()
		self.score_analyse()
		self.dist_mem = [[dist/self.win_diag for dist in self.sense_dist]]*self.loc_mem_sz
		self.state = np.append(np.array([self.dist_mem[i] for i in self.dist_mem_ind]).reshape(-1),[self.vel/self.vel_max])
		return self.state

	def step(self, action_in):
		# calculate movements
		dist = self.vel*self.time_per_frame
		drag = self.aero_drag_v2*self.vel**2
		total_grip_avail = (self.aero_down_v2*self.vel**2 + self.mass*9.81)*self.friction_coeff
		rotation = None
		if self.continuous:
			if len(action_in)==1:
				act_for_aft, act_steer = action_in[0], 0
			else:
				act_for_aft, act_steer = action_in
		else:
			action = self.actions_avail[action_in]
			if len(action)==1:
				act_for_aft, act_steer = action[0], 0
			else:
				act_for_aft, act_steer = action
		metrics = [self.lap_float, act_for_aft, act_steer, self.vel, total_grip_avail]
		long_force_max = min(self.forward_force, total_grip_avail) if act_for_aft >= 0 else -total_grip_avail
		long_force = abs(act_for_aft)*long_force_max
		self.vel = max(0,self.vel+(long_force - drag)/self.mass*self.time_per_frame)
		turn_grip_avail = (total_grip_avail**2-long_force**2)**0.5
		if act_steer != 0:
			rot_sign = np.sign(act_steer)
			r_turn_grip = np.inf if turn_grip_avail == 0 else self.mass*self.vel**2/turn_grip_avail
			if r_turn_grip == 0:
				steer_ang_grip = np.inf
			elif np.isinf(r_turn_grip):
				steer_ang_grip = 0
			else:
				steer_ang_grip = atan(self.wheelbase/r_turn_grip)
			steer_ang_max = min(self.steer_lock_ang, steer_ang_grip)
			steer_ang = abs(act_steer)*steer_ang_max
			if steer_ang != 0:
				rotation = (rot_sign, steer_ang)
		metrics = metrics + [long_force, steer_ang]

		# implement movements and update scores and statuses
		self.move(dist, rotation)
		self.frame_curr +=1
		self.score_analyse()
		reward = self.score-self.score_prev
		# reward = 1 if self.score>self.score_prev else 0
		if self.finished_course or not self.on_course or (self.frame_curr - self.lap_float_frame_max) >= self.patience:
			self.finished_episode = True
		# update sensing history
		del self.dist_mem[-1]
		self.dist_mem = [[dist/self.win_diag if dist is not None else 0 for dist in self.sense_dist]] + self.dist_mem
		self.state = np.append(np.array([self.dist_mem[i] for i in self.dist_mem_ind]).reshape(-1),[self.vel/self.vel_max])

		return self.state, reward, self.finished_episode, metrics

	def score_analyse(self):
		# calculate score
		self.score_prev = self.score
		self.score = self.lap_float
		# update maximum position
		if self.lap_float_max is None:
			new_max = True
		else:
			if self.lap_float > self.lap_float_max:
				new_max = True
			else:
				new_max = False
		if new_max:
			self.lap_float_max = self.lap_float
			self.lap_float_frame_max = self.frame_curr

	def move(self, dist, rotation):
		# save old track progress
		self.track_prog_prev = self.track_prog
		# move car
		if rotation is None:
			angle_rad = self.bear/180*pi
			x_delt = dist*sin(angle_rad)
			y_delt = dist*cos(angle_rad)
			self.hitbox_center = (self.hitbox_center[0]+x_delt, self.hitbox_center[1]+y_delt)
		else:
			# calculate center of rotation and angle or rotation
			rot_sign, steer_ang = rotation
			rot_radius = self.wheelbase/tan(steer_ang)
			angle_rad = (self.bear+90*rot_sign)/180*pi
			x_delt = rot_radius*sin(angle_rad)
			y_delt = rot_radius*cos(angle_rad)
			rot_center = (self.hitbox_center[0]+x_delt, self.hitbox_center[1]+y_delt)
			rot_ang_rad = dist/rot_radius*rot_sign
			# rotate car centre point and directional vector
			self.hitbox_center = rotate_points([self.hitbox_center], -rot_ang_rad, rot_center)[0]
			self.bear+=rot_ang_rad/pi*180
		self.hitbox_coords = self.outer_points()
		# analyse position and store in memory
		self.pos_analyse()
		del self.loc_mem[-1]
		self.loc_mem = [self.hitbox_center] + self.loc_mem

	def outer_points(self):
		# non rotated outer points
		xd, yd = self.sz[0]/2, self.sz[1]/2
		point_nonrot = [(self.hitbox_center[0]+x, self.hitbox_center[1]+y) for (x,y) in [(-xd,-yd), (xd,-yd), (xd,yd), (-xd,yd)]]
		# rotate outer points
		point_out = rotate_points(point_nonrot, -self.bear/180*pi, self.hitbox_center)
		return point_out

	def pos_analyse(self):
		# check on the course
		car_poly = Polygon(self.hitbox_coords)
		self.on_course = self.outer_poly.contains(car_poly) and not(self.inner_poly.intersects(car_poly))
		# progress round track
		car_loc = Point(self.hitbox_center)
		self.track_prog = self.middle_lin.project(car_loc, normalized = True)
		if self.track_prog_prev is not None:
			track_prog_change = self.track_prog - self.track_prog_prev
			if track_prog_change < -0.8:
				self.lap_whole +=1
			elif track_prog_change > 0.8:
				self.lap_whole -=1
		self.lap_float = min(self.lap_targ, self.lap_whole + self.track_prog)
		# location on track
		car_loc_track = self.middle_lin.interpolate(self.track_prog, normalized = True)
		temp = list(car_loc_track.coords)[0]
		self.track_loc = (temp[0], temp[1])
		# check finished course
		if self.lap_float is not None:
			self.finished_course = self.lap_float >= self.lap_targ
		# sense surroundings
		x,y = self.hitbox_center
		dist_min = []
		line_min = []
		dist = 1000
		for ang_sense in self.sense_ang:
			angle = (ang_sense+self.bear)/180*pi
			line_dir = LineString([(x, y), (x+dist*sin(angle), y+dist*cos(angle))])
			dist_angle_min = None
			line_angle_min = None
			track_lin_list = [self.outer_lin, self.inner_lin] 
			for track_lin in track_lin_list:
				difference = line_dir.difference(track_lin)
				if difference.geom_type == 'MultiLineString':
					line_curr = list(difference.geoms)[0]
					dist_curr = line_curr.length
					if dist_angle_min is None or dist_curr < dist_angle_min:
						line_angle_min, dist_angle_min = line_curr, dist_curr
			dist_min.append(dist_angle_min)
			if line_angle_min is None:
				line_min.append(line_angle_min)
			else:
				line_min.append([(x, y) for x,y in zip(*line_angle_min.xy)])
		self.sense_dist = dist_min
		self.sense_lin = line_min

	def render(self):
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(*self.win_sz)
			# track
			track_out = rendering.FilledPolygon(self.outer_coords)
			track_out.set_color(.5,.5,.5)
			self.viewer.add_geom(track_out)
			track_in = rendering.FilledPolygon(self.inner_coords)
			track_in.set_color(1,1,1)
			self.viewer.add_geom(track_in)
			# car - image
			car_img = rendering.Image("car_img.png", *self.sz)
			self.car_img_trans = rendering.Transform()
			car_img.add_attr(self.car_img_trans)
			self.viewer.add_geom(car_img)
			# car - box
			l,r,t,b = -self.sz[0]/2, self.sz[0]/2, self.sz[1]/2, -self.sz[1]/2
			car_box = rendering.make_polygon([(l,b), (l,t), (r,t), (r,b)], filled=False)
			car_box.set_color(1,0,0)
			self.car_box_trans = rendering.Transform()
			car_box.add_attr(self.car_box_trans)
			self.viewer.add_geom(car_box)
			# car - track location
			car_track_loc = rendering.make_circle(2)
			car_track_loc.set_color(1,0,0)
			self.car_track_loc_trans = rendering.Transform()
			car_track_loc.add_attr(self.car_track_loc_trans)
			self.viewer.add_geom(car_track_loc)
			# car - sensing points
			car_sense_loc_trans = []
			for __ in self.sense_lin:
				circle = rendering.make_circle(3)
				circle.set_color(0,1,0)
				circle_trans = rendering.Transform()
				circle.add_attr(circle_trans)
				car_sense_loc_trans.append(circle_trans)
				self.viewer.add_geom(circle)
			self.car_sense_loc_trans = car_sense_loc_trans
			# car - history points
			car_loc_mem_trans = []
			for i in range(len(self.loc_mem)):
				if i in self.dist_mem_ind:
					circle = rendering.make_circle(2)
					circle.set_color(0,1,0)
				else:
					circle = rendering.make_circle(1)
					circle.set_color(0,0,1)
				circle_trans = rendering.Transform()
				circle.add_attr(circle_trans)
				car_loc_mem_trans.append(circle_trans)
				self.viewer.add_geom(circle)
			self.car_loc_mem_trans = car_loc_mem_trans

		# update positions
		self.car_img_trans.set_translation(*self.hitbox_center)
		self.car_img_trans.set_rotation(-self.bear/180*pi)
		self.car_box_trans.set_translation(*self.hitbox_center)
		self.car_box_trans.set_rotation(-self.bear/180*pi)
		self.car_track_loc_trans.set_translation(*self.track_loc)
		for line, trans in zip(self.sense_lin, self.car_sense_loc_trans):
			loc = line[1] if line is not None else self.hitbox_center
			trans.set_translation(*loc)
		for loc, trans in zip(self.loc_mem, self.car_loc_mem_trans):
			trans.set_translation(*loc)

		return self.viewer.render(return_rgb_array = True)

# import time
# environment = CarGameEnv(True)
# action = [0,0]
# while not environment.finished_episode:
# 	state, __, __, __ = environment.step(action)
# 	if state[1]>0.2:
# 		action = [1.0, 0.0] 
# 	elif state[1]<=0.2 and state[3]>0.2:
# 		action = [-1.0, 0.0] 
# 	elif state[1]<=0.2 and state[3]<=0.2 and state[2]-state[0]<=0.1:
# 		action = [0.0, 0.0] 
# 	elif state[1]<=0.2 and state[3]<=0.2 and state[2]-state[0]>0.1:
# 		action = [0.0, 1.0] 
# 	print(state, action)
	# time.sleep(0.1)
	# environment.render()