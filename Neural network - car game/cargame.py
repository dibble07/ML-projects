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
	def __init__(self, player_sz, sense_ang, course, lap_targ, neural_network):
		self.hitbox = pygame.Rect(course.middle_coords[0], player_sz)
		self.hitbox.center = course.middle_coords[0]
		self.xvel = 5
		self.yvel = 5
		self.track_loc = course.middle_coords[0]
		self.track_prog = 0
		self.lap = 0
		self.lap_targ = lap_targ
		self.on_course = True
		self.started = False
		self.finished = False
		self.input_method = neural_network
		self.sense_ang = sense_ang
		self.sense(course)
		img = pygame.image.load("car.png")
		self.img = pygame.transform.scale(img, (self.hitbox.w,self.hitbox.h))
		self.colour = (random.randint(0,256), random.randint(0,256), random.randint(0,256))

	def draw(self, win):
		win.blit(self.img, (self.hitbox.x, self.hitbox.y))
		pygame.draw.rect(win, self.colour, self.hitbox, 2)
		if self.finished:
			font = pygame.font.SysFont('arial', 12, True)
			text = font.render('{0:.3f}'.format(self.score()), True, (255, 0, 0)) 
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
			self.time_start = pygame.time.get_ticks()
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
			self.time_dur = pygame.time.get_ticks() - self.time_start

	def score(self):
		if self.finished:
			time_dur = self.time_dur
		else:
			time_dur = pygame.time.get_ticks()-self.time_start
		score_out = -self.lap_float + time_dur/1000/60/60
		return score_out

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

def RedrawGameWindow(win, course, player_list):
	win.fill((0,0,0))
	course.draw(win)
	player_started_list = []
	for player in player_list:
		player.draw(win)
		if player.started:
			player_started_list.append(player)
	if len(player_started_list) > 0:
		player_best = player_list[np.argmin(np.array([player.score() for player in player_started_list]))]
		font = pygame.font.SysFont('arial', 18)
		if player_best.finished:
			time_curr = pygame.time.get_ticks()
			text = font.render('Time: {0:.2f}'.format(player_best.time_dur/1000), True, (255, 255, 255)) 
			win.blit(text, (5, 5))
			text = font.render('Laps: {0:.2f}'.format(player_best.lap_float), True, (255, 255, 255)) 
			win.blit(text, (5, 25)) 
		else:
			time_curr = pygame.time.get_ticks()
			text = font.render('Time: {0:.2f}'.format((time_curr-player_best.time_start)/1000), True, (255, 255, 255)) 
			win.blit(text, (5, 5))
			text = font.render('Laps: {0:.2f}'.format(player_best.lap_float), True, (255, 255, 255)) 
			win.blit(text, (5, 25))
	pygame.display.update()

def PlayGame(method_list, fin_pause, lap_targ):

	# initialise game
	pygame.init()
	win_sz = (600,400)
	win = pygame.display.set_mode(win_sz)
	clock = pygame.time.Clock()
	sense_angle = np.linspace(0, 2*pi, num=8, endpoint = False)

	# initialise components
	track_init = [(100, 200), (200, 100), (500,100), (500, 300), (200, 300)]
	track = course(track_init, 50)
	car_sz = (16, 32)
	car_list = [player(car_sz, sense_angle, track, lap_targ, neur_net) for neur_net in method_list]

	# game loop
	run = True
	while run:

		# maintain frame rate
		clock.tick(15)

		# move car
		for car in car_list:
			x_move = 0
			y_move = 0
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
				print("use the neural net")
				hey = car.input_method.activate(car.sense_dist)
				print(hey)
			if any([i != 0 for i in [x_move, y_move]]) and not(car.finished):
				car.move(x_move, y_move, track)

		# quit game if desired
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False

		if all([car.finished for car in car_list]):
			run = False
	            
		RedrawGameWindow(win, track, car_list)

	pygame.time.delay(fin_pause)
	pygame.quit()
	return [car.score() for car in car_list if car.input_method is not None]