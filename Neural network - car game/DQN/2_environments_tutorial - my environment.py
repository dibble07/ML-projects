import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from homegym import CardGameEnv, CarGameEnv
tf.compat.v1.enable_v2_behavior()

environment = suite_gym.load('CartPole-v0')
# environment = CarGameEnv()
environment = CardGameEnv()

print('action_spec:', environment.action_spec())
print('time_step_spec.observation:', environment.time_step_spec().observation)
print('time_step_spec.step_type:', environment.time_step_spec().step_type)
print('time_step_spec.discount:', environment.time_step_spec().discount)
print('time_step_spec.reward:', environment.time_step_spec().reward)

action = np.array(0, dtype=np.int32)
time_step = environment.reset()
print(time_step)
while not time_step.is_last():
  time_step = environment.step(action)
  print(time_step)

utils.validate_py_environment(environment, episodes=5)