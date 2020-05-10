from math import cos, sin, pi
import numpy as np
import random
from shapely.geometry import Polygon, Point, LinearRing, LineString
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class CardGameEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=0, name='observation')
    self._state = 21
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = 21
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      return self.reset()

    if action == 0:
      new_card = -4
      self._state += new_card
    elif action == 1:
      self._episode_ended = True

    if self._episode_ended or self._state <= 0:
      reward = -self._state if self._state > 0 else -21
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      reward = 0
      return ts.transition(np.array([self._state], dtype=np.int32), reward, discount=1.0)

class CarGameEnv(py_environment.PyEnvironment):

  def __init__(self):
    # specifications
    self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.float32, minimum=0, name='observation')
    # misc
    self.viewer = None
    self.win_sz = (416,416)
    self.win_diag = (self.win_sz[0]**2+self.win_sz[1]**2)**0.5
    # temp=[]
    # for rot in [0, -1, 1]:
    #   for trans in [1, 0, -1]:
    #     temp.append((trans, rot))
    temp = [(1,0), (0,1)]
    self.move_comb = temp
    self.patience = 10
    self.lap_targ = 0.5
    self.loc_mem_sz = 50
    # initialise track
    # track_coords = [(100, 300), (100, 100), (200,100), (200,200), (300, 200), (300, 100), (500, 100), (500, 200), (400, 300), (500, 400), (500, 500), (200, 500), (200, 400)]
    track_coords = [(100, 100), (100, 300), (300, 300), (300, 100), (100, 100)]
    track_width = 30
    self.middle_poly = Polygon(track_coords)
    self.middle_lin = LinearRing(self.middle_poly.exterior.coords)
    temp_coords = self.unique_wrap([(x, y) for x,y in zip(*self.middle_poly.exterior.xy)])
    self.middle_coords = temp_coords[::-1]
    temp_poly = Polygon(self.middle_poly.buffer(track_width*-1).exterior, [self.middle_lin])
    temp_coords = self.unique_wrap([(x, y) for x,y in zip(*temp_poly.exterior.xy)])
    self.inner_coords = temp_coords[::-1]
    self.inner_poly = Polygon(self.inner_coords)
    self.inner_lin = LinearRing(self.inner_poly.exterior.coords)
    temp_poly = Polygon(self.middle_poly.buffer(track_width).exterior, [self.middle_lin])
    temp_coords = self.unique_wrap([(x, y) for x,y in zip(*temp_poly.exterior.xy)])
    self.outer_coords = temp_coords[::-1]
    self.outer_poly = Polygon(self.outer_coords)
    self.outer_lin = LinearRing(self.outer_poly.exterior.coords)
    # initialise car
    # self.sense_ang = np.linspace(-90, 90, num=5, endpoint = True)
    self.sense_ang = np.linspace(0, 0, num=1, endpoint = True)
    self.bvel = -2
    self.fvel = 4
    self.rvel = 360/4
    self.sz = (16, 32)
    # reset
    self.reset()

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self.finished_episode = False
    self.frame_curr = 0
    self.bear = 0
    self.lap_whole = 0
    self.lap_float = None
    self.track_prog_prev = None
    self.score = None
    self.score_max = self.score
    self.score_max_frame = self.frame_curr
    self.hitbox_center = self.middle_coords[0]
    self.hitbox_coords = self.outer_points()
    self.loc_mem = [self.hitbox_center]*self.loc_mem_sz
    self.pos_analyse()
    self.score_analyse()

    obs = self.sense_dist[0]/max(*self.win_sz)*20 if self.sense_dist[0] is not None else 0
    return ts.restart(np.array([obs], dtype=np.float32))

  def _step(self, action):

    if self.finished_episode:
      return self.reset()

    if action == 1:
      self.finished_episode = True
      for_unit, bear_unit = 0, 0
    else:
      for_unit, bear_unit = self.move_comb[action]
    if bear_unit != 0 or for_unit != 0:
      self.move(bear_unit, for_unit)
    self.frame_curr +=1
    self.score_analyse()
    if self.finished_course or not self.on_course or (self.frame_curr - self.score_max_frame) >= self.patience:
      self.finished_episode = True

    if self.finished_episode:
      reward = -self.sense_dist[0]/max(*self.win_sz)*20 if self.on_course else -self.win_diag/max(*self.win_sz)*20
      obs = self.sense_dist[0]/max(*self.win_sz)*20 if self.on_course else 0
      return ts.termination(np.array([obs], dtype=np.float32), reward)
    else:
      reward = 0
      obs = self.sense_dist[0]/max(*self.win_sz)*20 if self.on_course else 0
      return ts.transition(np.array([obs], dtype=np.float32), reward, discount=1.0)

  def score_analyse(self):
    self.score_prev = self.score
    self.score = self.lap_float

    if self.score_max is None:
      new_max = True
    else:
      if self.score > self.score_max:
        new_max = True
      else:
        new_max = False
    if new_max:
      self.score_max = self.score
      self.score_max_frame = self.frame_curr

  def move(self, bear_unit, for_unit):
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
    x_delt = vel*sin(angle_rad)
    y_delt = vel*cos(angle_rad)
    self.hitbox_center = (self.hitbox_center[0]+x_delt, self.hitbox_center[1]+y_delt)
    self.hitbox_coords = self.outer_points()
    # analyse position and store in memory
    self.pos_analyse()
    del self.loc_mem[self.loc_mem_sz-1]
    self.loc_mem = [self.hitbox_center] + self.loc_mem

  def outer_points(self):
    # non rotated outer points
    xd, yd = self.sz[0]/2, self.sz[1]/2
    point_nonrot = [(self.hitbox_center[0]+x, self.hitbox_center[1]+y) for (x,y) in [(-xd,-yd), (xd,-yd), (xd,yd), (-xd,yd)]]
    # rotate outer points
    angle_rad = self.bear/180*pi
    ox, oy = self.hitbox_center
    point_out = []
    for point in point_nonrot:
      px, py = point
      cs = cos(angle_rad)
      sn = sin(angle_rad)
      xd = px - ox
      yd = py - oy
      qx = ox + cs * xd - sn * yd
      qy = oy + sn * xd + cs * yd
      point_out.append((qx, qy))
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
    self.lap_float = self.lap_whole + self.track_prog
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

  def unique_wrap(self, list_in):
    out=[]
    for i in list_in:
      if i not in out:
        out.append(i)
    out.append(out[0])
    return out

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
      for __ in self.loc_mem:
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
