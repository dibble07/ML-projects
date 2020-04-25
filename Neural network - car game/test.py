from math import cos, sin, pi
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LinearRing, LineString
import numpy as np

# initialise track centreline and car_loc
track_init = [(0, 0), (1000,0), (1000, 500), (0, 500)]
track_mid_poly = Polygon(track_init)
track_mid_ext = LinearRing(track_mid_poly.exterior.coords)
car_loc = Point(200, 50)

# plot track bounds
plt.figure()
track_inner_poly = Polygon(track_mid_poly.buffer(-100).exterior, [track_mid_ext])
track_inner_ext = LinearRing(track_inner_poly.exterior.coords)
track_outer_poly = Polygon(track_mid_poly.buffer(100).exterior, [track_mid_ext])
track_outer_coords = [(np.round(x), np.round(y)) for x,y in zip(*track_outer_poly.exterior.xy)]
track_outer_poly = Polygon(track_outer_coords)
track_outer_ext = LinearRing(track_outer_poly.exterior.coords)
plt.plot(*track_mid_poly.exterior.xy)
plt.plot(*track_outer_poly.exterior.xy)
plt.plot(*track_inner_poly.exterior.xy)

# point on track midline nearest to car
plt.plot(*car_loc.xy,'ro')
pol_ext = LinearRing(track_mid_poly.exterior.coords)
car_loc_track = track_mid_ext.interpolate(track_mid_ext.project(car_loc))
plt.plot(*car_loc_track.xy,'bo')
closest_point_coords = list(car_loc_track.coords)[0]

# nearest track in each direction
x,y = car_loc.x, car_loc.y
print(range(0,360,45))
for angle_deg in range(0,360,45):
	angle_rad = angle_deg * pi / 180.0 
	dist = 1000 
	line_dir = LineString([(x, y), (x + dist * sin(angle_rad), y + dist * cos(angle_rad))])
	dist_min = None  
	line_min = None 
	track_lin_list = [track_outer_ext, track_inner_ext] 
	for track_lin in track_lin_list:
		difference = line_dir.difference(track_lin)
		if difference.geom_type == 'MultiLineString':
			line_curr = list(difference.geoms)[0]
			dist_curr = line_curr.length
			if dist_min is None or dist_curr < dist_min:
				line_min, dist_min = line_curr, dist_curr
	print(dist_min)
	plt.plot(*line_min.xy)



plt.show()