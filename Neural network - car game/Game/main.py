import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pygame
pygame.init()
from shapely.geometry import Polygon, Point, LinearRing

def UniqueWrap(list_in):
	out=[]
	for i in list_in:
		if i not in out:
			out.append(i)
	out.append(out[0])
	return out

def RedrawGameWindow():
	win.fill((0,0,0))
	if not(car.first_move):
		font = pygame.font.SysFont('arial', 18)
		if car.finished:
			time_curr = pygame.time.get_ticks()
			text = font.render('Time: {0:.2f}'.format(car.time_dur/1000), True, (255, 255, 255)) 
			win.blit(text, (5, 5))
			text = font.render('Laps: {0}'.format(car.lap), True, (255, 255, 255)) 
			win.blit(text, (5, 25)) 
		else:
			time_curr = pygame.time.get_ticks()
			text = font.render('Time: {0:.2f}'.format((time_curr-car.time_start)/1000), True, (255, 255, 255)) 
			win.blit(text, (5, 5))
			text = font.render('Laps: {0}'.format(car.lap), True, (255, 255, 255)) 
			win.blit(text, (5, 25)) 
	track.draw(win)
	car.draw(win)
	pygame.display.update()

class course(object):
	def __init__(self,coords,track_width):
		self.middle_poly = Polygon(coords)
		self.middle_lin = LinearRing(self.middle_poly.exterior.coords)
		temp_poly = Polygon(self.middle_poly.buffer(track_width*-1).exterior, [self.middle_lin])
		temp_coords = UniqueWrap([(int(np.round(x)), int(np.round(y))) for x,y in zip(*temp_poly.exterior.xy)])
		self.inner_coords = temp_coords[::-1]
		self.inner_poly = Polygon(self.inner_coords)
		temp_poly = Polygon(self.middle_poly.buffer(track_width).exterior, [self.middle_lin])
		temp_coords = UniqueWrap([(int(np.round(x)), int(np.round(y))) for x,y in zip(*temp_poly.exterior.xy)])
		self.outer_coords = temp_coords[::-1]
		self.outer_poly = Polygon(self.outer_coords)

	def draw(self, win):
		pygame.draw.polygon(win, (128,128,128), self.outer_coords)
		pygame.draw.polygon(win, (0,0,0), self.inner_coords)

class player(object):
	def __init__(self, x, y, width, height):
		self.hitbox = pygame.Rect(x, y, width, height)
		self.xvel = 5
		self.yvel = 5
		self.track_loc = (0, 0)
		self.track_prog = 0
		self.track_prog_prev = 0
		self.lap = 0
		self.on_course = True
		self.first_move = True
		self.finished = False
		img = pygame.image.load("car.png")
		self.img = pygame.transform.scale(img, (self.hitbox.w,self.hitbox.h))

	def draw(self, win):
		win.blit(self.img, (self.hitbox.x, self.hitbox.y))
		pygame.draw.rect(win, (255,0,0), self.hitbox,2)
		pygame.draw.circle(win, (0, 255, 0), self.track_loc, 5)

	def move(self, x_unit, y_unit, course):
		# save start time
		if self.first_move:
			self.first_move = False
			self.time_start = pygame.time.get_ticks()
		# save old track progress
		self.track_prog_prev = self.track_prog
		# move car
		x_delt = self.xvel*x_unit
		y_delt = self.yvel*y_unit
		self.hitbox.move_ip(x_delt,y_delt)
		# progress round track
		car_loc = Point(self.hitbox.center)
		self.track_prog = course.middle_lin.project(car_loc, normalized = True)
		track_prog_change = self.track_prog - self.track_prog_prev
		if track_prog_change < -0.8:
			self.lap +=1
		elif track_prog_change > 0.8:
			self.lap -=1
		if self.lap >= lap_targ:
			self.score_final = self.score()
			self.finished = True
		# location on track
		car_loc_track = course.middle_lin.interpolate(self.track_prog, normalized = True)
		temp = list(car_loc_track.coords)[0]
		self.track_loc = (int(np.round(temp[0])), int(np.round(temp[1])))
		# check on course
		car_poly = Polygon([self.hitbox.topleft, self.hitbox.bottomleft, self.hitbox.topright, self.hitbox.bottomright])
		self.on_course = course.outer_poly.contains(car_poly) and not(course.inner_poly.intersects(car_poly))

	def score(self):
		if self.finished:
			score_out = self.score_final
		else:
			lap_float = self.lap + self.track_prog
			time_curr = pygame.time.get_ticks()
			time_dur = time_curr-car.time_start
			score_out = -lap_float + time_dur/1000/60/60
			if self.lap >= lap_targ:
				self.finished = True
				self.score_final = score_out
				self.time_dur = time_dur
		return score_out


# initialise game
win_sz = (600,400)
win = pygame.display.set_mode(win_sz)
clock = pygame.time.Clock()

# initialise components
track_init = [(100, 200), (200, 100), (500,100), (500, 300), (200, 300)]
track = course(track_init, 50)
car = player(100-16/2, 200-32/2, 16, 32)

# define parameters
lap_targ = 2

run = True
while run:

	# clock something
	clock.tick(27)

	# move car
	keys = pygame.key.get_pressed()
	x_move = 0
	y_move = 0
	if keys[pygame.K_LEFT]:
		x_move = -1
	elif keys[pygame.K_RIGHT]:
		x_move = 1
	if keys[pygame.K_UP]:
		y_move = -1
	elif keys[pygame.K_DOWN]:
		y_move = 1
	if any([i != 0 for i in [x_move, y_move]]) and car.on_course and not(car.finished):
		car.move(x_move, y_move, track)

	# quit game if car off course
	# if not(car.on_course):
	# 	pygame.time.wait(500)
	# 	run = False

	# quit game if desired
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False
            
	RedrawGameWindow()

pygame.quit()