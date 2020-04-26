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
		self.hitbox = pygame.Rect(course.middle_coords[0], player_sz)
		self.hitbox.center = course.middle_coords[0]
		self.xvel = 5
		self.yvel = 5
		self.track_loc = course.middle_coords[0]
		self.track_prog = 0
		self.lap = 0
		self.lap_targ = lap_targ
		self.lap_float = min(self.lap + self.track_prog, self.lap_targ)
		self.on_course = True
		self.started = False
		self.time_init = time_init
		self.time_dur = 0
		self.finished = False
		self.input_method = neural_network
		self.sense_ang = sense_ang
		self.sense(course)
		img = pygame.image.load("car.png")
		self.img = pygame.transform.scale(img, (self.hitbox.w,self.hitbox.h))
		self.colour = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

	def draw(self, win):
		win.blit(self.img, (self.hitbox.x, self.hitbox.y))
		pygame.draw.rect(win, self.colour, self.hitbox, 2)
		if self.finished:
			font = pygame.font.SysFont('arial', 12, True)
			text = font.render('{0:.3f}'.format(self.score), True, (255, 0, 0)) 
			win.blit(text, self.hitbox.center)
		else:
			for line in self.sense_lin:
				if line is not None:
					pygame.draw.line(win, self.colour, line[0], line[1], 2)
		pygame.draw.circle(win, self.colour, self.track_loc, 5)

	def move(self, x_unit, y_unit, course):
		# save start time
		if not(self.started):
			self.started = True
			self.time_start = pygame.time.get_ticks() - self.time_init
		# save old track progress
		self.track_prog_prev = self.track_prog
		# move car
		x_delt = self.xvel*x_unit
		y_delt = self.yvel*y_unit
		self.hitbox.move_ip(x_delt,y_delt)
		# sense surroundings
		self.sense(course)
		# progress round track
		car_loc = Point(self.hitbox.center)
		self.track_prog = course.middle_lin.project(car_loc, normalized = True)
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
		# check finished
		car_poly = Polygon([self.hitbox.topleft, self.hitbox.bottomleft, self.hitbox.topright, self.hitbox.bottomright])
		self.on_course = course.outer_poly.contains(car_poly) and not(course.inner_poly.intersects(car_poly))
		if self.lap_float == self.lap_targ or not(self.on_course):
			self.finished = True

	def frame(self, time_curr, patience):
		if self.started:
			self.time_dur = time_curr - self.time_start
		else:
			self.time_dur = 0
		self.score = self.lap_float - self.lap_targ - self.time_dur/1000/60/60
		if time_curr/1000 >= patience:
			self.finished = True

	def sense(self, course):
		x,y = self.hitbox.center
		dist_min = []
		line_min = []
		dist = 1000
		for angle in self.sense_ang:
			line_dir = LineString([(x, y), (x+dist*sin(angle), y+dist*cos(angle))])
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

def UniqueWrap(list_in):
	out=[]
	for i in list_in:
		if i not in out:
			out.append(i)
	out.append(out[0])
	return out

def RedrawGameWindow(win, course, player_list, time_curr):
	win.fill((0,0,0))
	course.draw(win)
	for player in player_list:
		player.draw(win)
	max_lap = max([player.lap_float for player in player_list])
	font = pygame.font.SysFont('arial', 18)
	text = font.render('Time: {0:.2f}'.format(time_curr/1000), True, (255, 255, 255)) 
	win.blit(text, (5, 5))
	text = font.render('Laps: {0:.2f}'.format(max_lap), True, (255, 255, 255)) 
	win.blit(text, (5, 25)) 
	pygame.display.update()

def PlayGame(method_list, fin_pause, lap_targ, patience):

	# initialise game
	pygame.init()
	win_sz = (600,400)
	win = pygame.display.set_mode(win_sz)
	clock = pygame.time.Clock()
	sense_angle = np.linspace(0, 2*pi, num=8, endpoint = False)
	time_init = pygame.time.get_ticks()

	# initialise components
	track_init = [(100, 200), (200, 100), (500,100), (500, 300), (200, 300)]
	track = course(track_init, 50)
	car_sz = (16, 32)
	car_list = [player(car_sz, sense_angle, track, lap_targ, neur_net, time_init) for neur_net in method_list]

	# game loop
	run = True
	while run:

		# maintain frame rate
		clock.tick(20)

		# move cars
		for car in car_list:
			x_move = 0
			y_move = 0
			# get keyboard input
			if car.input_method is None:
				keys = pygame.key.get_pressed()
				if keys[pygame.K_LEFT]:
					x_move = -1
				elif keys[pygame.K_RIGHT]:
					x_move = 1
				if keys[pygame.K_UP]:
					y_move = -1
				elif keys[pygame.K_DOWN]:
					y_move = 1
			else:
				# get neural netowrk input
				neur_net_input = [dist / sum(win_sz) for dist in car.sense_dist]
				pred_float = car.input_method.activate(neur_net_input)
				if pred_float[2] >= 0:
					x_move = -1
				elif pred_float[3] >= 0:
					x_move = 1
				if pred_float[0] >= 0:
					y_move = -1
				elif pred_float[1] >= 0:
					y_move = 1
			# move car
			if any([i != 0 for i in [x_move, y_move]]) and not(car.finished):
				car.move(x_move, y_move, track)

		# update time and score
		time_frame = pygame.time.get_ticks() - time_init
		for car in car_list:
			if not(car.finished):
				car.frame(time_frame, patience)

		# check if any cars are not finished
		if all([car.finished for car in car_list]):
			run = False

		# quit game if desired
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
	            
		RedrawGameWindow(win, track, car_list, time_frame)

	pygame.time.delay(fin_pause)
	pygame.quit()
	return [car.score for car in car_list if car.input_method is not None]