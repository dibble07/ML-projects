import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LinearRing
import numpy as np

# initialise track centreline
track_init = [(0, 0), (1000,0), (1000, 500), (0, 500)]
track_mid_poly = Polygon(track_init)
track_mid_ext = LinearRing(track_mid_poly.exterior.coords)

# plot track bounds
plt.figure()
track_outer_poly = Polygon(track_mid_poly.buffer(100).exterior, [track_mid_ext])
track_inner_poly = Polygon(track_mid_poly.buffer(-100).exterior, [track_mid_ext])
print(track_outer_poly.exterior.xy)
track_outer_coords = [(np.round(x), np.round(y)) for x,y in zip(*track_outer_poly.exterior.xy)]
track_outer_poly = Polygon(track_outer_coords)
track_outer_ext = LinearRing(track_outer_poly.exterior.coords)
print(track_outer_coords)
print(list(set(track_outer_coords)) )
plt.plot(*track_mid_poly.exterior.xy)
plt.plot(*track_outer_poly.exterior.xy)
plt.plot(*track_inner_poly.exterior.xy)

# point on polygon nearest to point
car_loc = Point(4.5, -0.5)
plt.plot(*car_loc.xy,'ro')
pol_ext = LinearRing(track_mid_poly.exterior.coords)
car_loc_track = track_mid_ext.interpolate(track_mid_ext.project(car_loc))
plt.plot(*car_loc_track.xy,'bo')
closest_point_coords = list(car_loc_track.coords)[0]


plt.figure()
track = Polygon([(0, 0), (10,0), (10, 5), (0, 5)])
x,y = track.exterior.xy
plt.plot(x,y)
car1 = Polygon([(1, 1), (9,1), (9, 4), (1, 4)])
x,y = car1.exterior.xy
plt.plot(x,y)
print(track.contains(car1))
car2 = Polygon([(5, 0), (15,0), (15, 5), (5, 5)])
x,y = car2.exterior.xy
plt.plot(x,y)
print(track.intersects(car2))




plt.show()