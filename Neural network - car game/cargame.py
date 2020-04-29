from math import cos, sin, pi
import numpy as np
import pygame
import random
from shapely.geometry import Polygon, Point, LinearRing, LineString

class course(object):
	def __init__(self, coords, track_width):
		# middle
		self.middle_poly = Polygon(coords)
		self.middle_lin = LinearRing(self.middle_poly.exterior.coords)
		temp_coords = UniqueWrap([(int(np.round(x)), int(np.round(y))) for x,y in zip(*self.middle_poly.exterior.xy)])
		self.middle_coords = temp_coords[::-1]
		# inner
		temp_poly = Polygon(self.middle_poly.buffer(track_width*-1).exterior, [self.middle_lin])
		temp_coords = UniqueWrap([(int(np.round(x)), int(np.round(y))) for x,y in zip(*temp_poly.exterior.xy)])
		self.inner_coords = temp_coords[::-1]
		self.inner_poly = Polygon(self.inner_coords)
		self.inner_lin = LinearRing(self.inner_poly.exterior.coords)
		# outer
		temp_poly = Polygon(self.middle_poly.buffer(track_width).exterior, [self.middle_lin])
		temp_coords = UniqueWrap([(int(np.round(x)), int(np.round(y))) for x,y in zip(*temp_poly.exterior.xy)])
		self.outer_coords = temp_coords[::-1]
		self.outer_poly = Polygon(self.outer_coords)
		self.outer_lin = LinearRing(self.outer_poly.exterior.coords)

	def draw(self, win):
		pygame.draw.polygon(win, (128,128,128), self.outer_coords)
		pygame.draw.polygon(win, (0,0,0), self.inner_coords)

class player(object):
	def __init__(self, player_sz, sense_ang, course, lap_targ, neural_network, time_init):
		self.finished = False
		self.started = False
		self.input_method = neural_network
		self.lap_targ = lap_targ
		self.sense_ang = sense_ang
		self.time_init = time_init
		self.bear = 0
		self.bvel = -2
		self.fvel = 5
		self.lap = 0
		self.rvel = 360/36
		self.colour = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
		img = pygame.image.load("car_img.png")
		self.img = pygame.transform.scale(pygame.transform.rotate(img, 180), (player_sz[0],player_sz[1]))
		self.hitbox_center = course.middle_coords[0]
		xd = int(np.ceil(player_sz[0]/2))
		yd = int(np.ceil(player_sz[1]/2))
		self.hitbox_coords_nonrot = [(self.hitbox_center[0]+x, self.hitbox_center[1]+y) for (x,y) in [(-xd,-yd), (xd,-yd), (xd,yd), (-xd,yd)]]
		self.hitbox_coords = RotatePoints(self.hitbox_center, self.hitbox_coords_nonrot, self.bear)
		self.pos_analyse(course)

	def draw(self, win):
		temp = pygame.transform.rotate(self.img, -self.bear)
		xd = int(np.ceil(temp.get_width()/2))
		yd = int(np.ceil(temp.get_height()/2))
		win.blit(temp, (self.hitbox_center[0]-xd, self.hitbox_center[1]-yd))
		pygame.draw.polygon(win, self.colour, self.hitbox_coords, 2)
		if self.finished:
			font = pygame.font.SysFont('arial', 12, True)
			text = font.render('{0:.3f}'.format(self.score), True, (255, 0, 0)) 
			win.blit(text, self.hitbox_center)
		else:
			for line in self.sense_lin:
				if line is not None:
					pygame.draw.line(win, self.colour, line[0], line[1], 2)
		pygame.draw.circle(win, self.colour, self.track_loc, 5)

	def move(self, bear_unit, for_unit, course):
		# save start time
		if not(self.started):
			self.started = True
			self.time_start = pygame.time.get_ticks() - self.time_init
		# save old track progress
		self.track_prog_prev = self.track_prog
		# move car
		self.bear+=bear_unit*self.rvel
		angle_rad = self.bear/180*pi
		if for_unit > 0:
			vel = self.fvel
		elif for_unit < 0:
			vel = self.bvel
		elif for_unit == 0:
			vel = 0
		x_delt = int(np.round(vel*sin(angle_rad)))
		y_delt = -int(np.round(vel*cos(angle_rad)))
		self.hitbox_center = (self.hitbox_center[0]+x_delt, self.hitbox_center[1]+y_delt)
		self.hitbox_coords_nonrot = [(x+x_delt, y+y_delt) for (x,y) in self.hitbox_coords_nonrot]
		self.hitbox_coords = RotatePoints(self.hitbox_center, self.hitbox_coords_nonrot, self.bear)
		# analyse position
		self.pos_analyse(course)
		# check finished
		car_poly = Polygon(self.hitbox_coords)
		self.on_course = course.outer_poly.contains(car_poly) and not(course.inner_poly.intersects(car_poly))
		if self.lap_float == self.lap_targ or not(self.on_course):
			self.finished = True

	def frame(self, time_curr, patience):
		if self.started:
			self.time_dur = time_curr - self.time_start
		else:
			self.time_dur = 0
		self.score = self.lap_float - self.lap_targ# - self.time_dur/1000/60/self.lap_targ
		if patience is not None:
			if not(hasattr(self, 'score_max')):
					self.score_max = self.score
					self.score_max_time = time_curr
			if self.score > self.score_max:
				self.score_max = self.score
				self.score_max_time = time_curr
			elif (time_curr - self.score_max_time)/1000 >= patience:
				self.finished = True

	def pos_analyse(self, course):
		# sense surroundings
		x,y = self.hitbox_center
		dist_min = []
		line_min = []
		dist = 1000
		for ang_sense in self.sense_ang:
			angle = (ang_sense+self.bear)/180*pi
			line_dir = LineString([(x, y), (x+dist*sin(angle), y-dist*cos(angle))])
			dist_angle_min = None
			line_angle_min = None
			track_lin_list = [course.outer_lin, course.inner_lin] 
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
				line_min.append([(int(np.round(x)), int(np.round(y))) for x,y in zip(*line_angle_min.xy)])
		self.sense_dist = dist_min
		self.sense_lin = line_min
		# progress round track
		car_loc = Point(self.hitbox_center)
		self.track_prog = course.middle_lin.project(car_loc, normalized = True)
		if self.started:
			track_prog_change = self.track_prog - self.track_prog_prev
			if track_prog_change < -0.8:
				self.lap +=1
			elif track_prog_change > 0.8:
				self.lap -=1
		self.lap_float = min(self.lap + self.track_prog, self.lap_targ)
		# location on track
		car_loc_track = course.middle_lin.interpolate(self.track_prog, normalized = True)
		temp = list(car_loc_track.coords)[0]
		self.track_loc = (int(np.round(temp[0])), int(np.round(temp[1])))

def UniqueWrap(list_in):
	out=[]
	for i in list_in:
		if i not in out:
			out.append(i)
	out.append(out[0])
	return out

def RotatePoints(origin, point_in, angle_deg):
	angle_rad = angle_deg/180*pi
	ox, oy = origin
	point_out = []
	for point in point_in:
		px, py = point
		cs = cos(angle_rad)
		sn = sin(angle_rad)
		xd = px - ox
		yd = -(py - oy)
		qx = ox + cs * xd - sn * yd
		qy = oy + sn * xd + cs * yd
		point_out.append((int(np.round(qx)), int(np.round(qy))))
	return point_out

def RedrawGameWindow(win, course, player_list, time_curr):
	win.fill((0,0,0))
	course.draw(win)
	for player in player_list:
		player.draw(win)
	if all([player.finished for player in player_list]):
		player_best = player_list[np.argmax([player.score for player in player_list])]
		lap_print = player_best.lap_float
		time_print = player_best.time_dur
	else:
		lap_print = max([player.lap_float for player in player_list])
		time_print = time_curr
	font = pygame.font.SysFont('arial', 18)
	text = font.render('Time: {0:.2f}'.format(time_print/1000), True, (255, 255, 255)) 
	win.blit(text, (5, 5))
	text = font.render('Laps: {0:.2f}'.format(lap_print), True, (255, 255, 255)) 
	win.blit(text, (5, 25)) 
	pygame.display.update()

def PlayGame(method_list, win_str, fin_pause, lap_targ, patience):

	# initialise game
	pygame.init()
	pygame.display.set_caption(win_str)
	win_sz = (600,600)
	win = pygame.display.set_mode(win_sz)
	clock = pygame.time.Clock()
	sense_angle = np.linspace(-90, 90, num=5, endpoint = True)
	time_init = pygame.time.get_ticks()

	# initialise components
	track_init = [(100, 300), (100, 100), (200,100), (200,200), (300, 200), (300, 100), (500, 100), (500, 200), (400, 300), (500, 400), (500, 500), (200, 500), (200, 400)]
	track = course(track_init, 35)
	car_sz = (16, 32)
	car_list = [player(car_sz, sense_angle, track, lap_targ, neur_net, time_init) for neur_net in method_list]

	# game loop
	run = True
	while run:

		# maintain frame rate
		clock.tick(15)

		# move cars
		for car in car_list:
			bear_move = 0
			for_move = 0
			# get keyboard input
			if car.input_method is None:
				keys = pygame.key.get_pressed()
				if keys[pygame.K_LEFT]:
					bear_move-=1
				if keys[pygame.K_RIGHT]:
					bear_move+=1
				if keys[pygame.K_UP]:
					for_move+=1
				if keys[pygame.K_DOWN]:
					for_move-=1
			else:
				# get neural netowrk input
				neur_net_input = [dist/sum(win_sz) for dist in car.sense_dist]# + [car.track_prog/lap_targ]
				pred_float = car.input_method.activate(neur_net_input)
				if pred_float[2] >= 0:
					bear_move-=1
				if pred_float[3] >= 0:
					bear_move+=1
				if pred_float[0] >= 0:
					for_move+=1
				if pred_float[1] >= 0:
					for_move-=1
			# move car
			if any([i != 0 for i in [bear_move, for_move]]) and not(car.finished):
				car.move(bear_move, for_move, track)

		# update time and score
		time_frame = pygame.time.get_ticks() - time_init
		for car in car_list:
			if not(car.finished):
				car.frame(time_frame, patience)

		# check if any cars are not finished
		if all([car.finished for car in car_list]) and fin_pause is not None:
			run = False

		# quit game if desired
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
	            
		RedrawGameWindow(win, track, car_list, time_frame)

	if fin_pause is not None:
		pygame.time.delay(fin_pause)
	pygame.quit()
	return [car.score for car in car_list if car.input_method is not None]

# PlayGame([None], 'Test', 1000, 1, 5)