import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pygame
pygame.init()
from shapely.geometry import Polygon, Point, LinearRing

# initialise game
win_sz = (600,400)
win = pygame.display.set_mode(win_sz)
clock = pygame.time.Clock()
score = 0

def UniqueWrap(list_in):
	out=[]
	for i in list_in:
		if i not in out:
			out.append(i)
	out.append(out[0])
	return out

def RedrawGameWindow():
    pygame.draw.rect(win, (0,0,0), (0, 0, win_sz[0], win_sz[1]), 0)
    track.draw(win)
    car.draw(win)
    pygame.display.update()

class course(object):
	def __init__(self,coords,track_width):
		self.middle_poly = Polygon(coords)
		self.middle_lin = LinearRing(self.middle_poly.exterior.coords)
		self.middle_coords = [(int(np.round(x)), int(np.round(y))) for x,y in zip(*self.middle_poly.exterior.xy)]
		temp_poly = Polygon(self.middle_poly.buffer(track_width*-1).exterior, [self.middle_lin])
		temp_coords = UniqueWrap([(int(np.round(x)), int(np.round(y))) for x,y in zip(*temp_poly.exterior.xy)])
		self.inner_coords = temp_coords[::-1]
		self.inner_poly = Polygon(self.inner_coords)
		self.inner_lin = LinearRing(self.inner_poly.exterior.coords)
		temp_poly = Polygon(self.middle_poly.buffer(track_width).exterior, [self.middle_lin])
		temp_coords = UniqueWrap([(int(np.round(x)), int(np.round(y))) for x,y in zip(*temp_poly.exterior.xy)])
		self.outer_coords = temp_coords[::-1]
		self.outer_poly = Polygon(self.outer_coords)
		self.outer_lin = LinearRing(self.outer_poly.exterior.coords)

	def draw(self, win):
		pygame.draw.polygon(win, (128,128,128), self.outer_coords)
		pygame.draw.polygon(win, (0,0,0), self.inner_coords)

class player(object):
	def __init__(self, x, y, width, height):
		self.hitbox = pygame.Rect(x, y, width, height)
		self.xvel = 5
		self.yvel = 5
		self.track_loc = (0, 0)
		self.score = 0
		self.functioning = True
		self.img = pygame.transform.scale(pygame.image.load("car.png") , (self.hitbox.w,self.hitbox.h))

	def draw(self, win):
		win.blit(self.img, (self.hitbox.x, self.hitbox.y))
		pygame.draw.rect(win, (255,0,0), self.hitbox,2)
		pygame.draw.circle(win, (0, 255, 0), self.track_loc, 5)

	def on_course(self, course):
		poly = Polygon([self.hitbox.topleft, self.hitbox.bottomleft, self.hitbox.topright, self.hitbox.bottomright])
		out = course.outer_poly.contains(poly) and not(course.inner_poly.intersects(poly))
		return out

	def move(self, x_unit, y_unit, course):
		# move car
		x_delt = self.xvel*x_unit
		y_delt = self.yvel*y_unit
		self.hitbox.move_ip(x_delt,y_delt)
		# check on course
		if self.on_course(course):
			self.functioning = True
		else:
			self.functioning = False
		# location along track
		car_loc = Point(self.hitbox.center)
		car_loc_track = course.middle_lin.interpolate(course.middle_lin.project(car_loc))
		print("")
		print(course.middle_lin.project(car_loc))
		print(car_loc_track)
		temp = list(car_loc_track.coords)[0]
		self.track_loc = (int(np.round(temp[0])), int(np.round(temp[1])))
		# pause if leave track
		if not(self.functioning):
			hit_quit = False
			for i in range(10):
				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						hit_quit = True
				if hit_quit:
					pygame.quit()
				else:
					pygame.time.delay(100)

track_init = [(100, 200), (200, 100), (500,100), (500, 300), (200, 300)]
track = course(track_init, 50)
car = player(100, 200, 16, 32)

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
	car.move(x_move, y_move, track)

	# quit game if desired
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False
            
	RedrawGameWindow()

pygame.quit()