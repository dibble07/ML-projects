# Import libraries
from cpprb import ReplayBuffer, PrioritizedReplayBuffer
from datetime import datetime
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import imageio
from math import pi
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
matplotlib.rcParams.update({'font.size': 6})
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import tensorflow as tf

from algos import DQN, DDPG
from homegym import CarGameEnv

# Define user functions
def evaluate_policy(show):
	render_frames=[]
	metrics_dict = {"laps": [], "act_long": [], "act_steer": [], "act_head_rot": [], "head_rot_ang": [], "speed": [],
	 "grip": [], "longitudinal_force": [], "steering_angle": []}
	obs = test_env.reset()
	done = False
	while not done:
		action = agent.get_action(obs, test=True)
		next_obs, _, done, (laps, act_long, act_steer, speed, grip, longitudinal_force, steering_angle) = test_env.step(action)
		if show:
			render_frames.append(test_env.render())
			metrics_dict["laps"].append(laps)
			metrics_dict["act_long"].append(act_long)
			metrics_dict["act_steer"].append(act_steer)
			metrics_dict["speed"].append(speed)
			metrics_dict["grip"].append(grip)
			metrics_dict["longitudinal_force"].append(longitudinal_force)
			metrics_dict["steering_angle"].append(steering_angle)
		obs = next_obs
	test_env.close()
	return test_env.lap_float, test_env.frame_curr, render_frames, metrics_dict

def display_episode(render, metrics, filedir):
	# initialise figure
	fig1, ax1 = plt.subplots(nrows=4, figsize=(8,5))
	# plot metrics
	ax1[0].plot(metrics["laps"],metrics["act_long"], label="Fore-aft")
	ax1[0].plot(metrics["laps"],metrics["act_steer"], label="Steer")
	ax1[0].set_ylabel("Actions [-]")
	ax1[0].legend(loc="best")
	ax1[1].plot(metrics["laps"],[x/1000 for x in metrics["grip"]], label="Grip")
	ax1[1].plot(metrics["laps"],[x/1000 if x>0 else 0 for x in metrics["longitudinal_force"]], label="Acceleration")
	ax1[1].plot(metrics["laps"],[-x/1000 if x<0 else 0 for x in metrics["longitudinal_force"]], label="Braking")
	ax1[1].set_ylabel("Forces [kN]")
	ax1[1].legend(loc="best")
	ax1[2].plot(metrics["laps"],[x*180/pi for x in metrics["steering_angle"]])
	ax1[2].set_ylabel("Steer angle [deg]")
	ax1[3].plot(metrics["laps"],metrics["speed"])
	ax1[3].set_ylabel("Speed [m/s]")
	fig1.tight_layout()
	fig1.savefig(filedir + "/metrics.png")

	# save video without metrics
	with imageio.get_writer(filedir + "/video.mp4", fps=30) as video:
		for frame in render:
			video.append_data(frame)

# Define variables
continuous = True
use_prioritized_rb=True
max_steps=500_000
test_interval=500
memory_capacity=100_000
batch_size=64

# Initialise environment, policy and replay buffer
env = CarGameEnv(continuous)
test_env = CarGameEnv(continuous)
if continuous:
	agent = DDPG(
		state_shape=env.observation_space.shape,
		action_dim=env.action_space.high.size,
		discount=0.99,
		load_model=None,
		actor_units=[64, 32],
		critic_units=[64, 32],
		sigma=0.1,
		tau=0.005,
		max_action=env.action_space.high[0]
		)
else:
	agent = DQN(
		state_shape=env.observation_space.shape,
		action_dim=env.action_space.n,
		discount=0.99,
		load_model=None,
		units=[16, 8],
		enable_dueling_dqn=True,
		target_replace_interval=300,
		epsilon_init_step = 100_000,
		epsilon_init = 0.1,
		epsilon_final = 0.01,
		epsilon_decay = 0.0001
		)
obs_space_shape = env.observation_space.shape
act_space_shape = env.action_space.shape if continuous else [1, ]
env_dict={
"obs": {"shape": obs_space_shape},
"next_obs": {"shape": obs_space_shape},
"act": {"shape": act_space_shape},
"rew": {},
"done": {}
}
if use_prioritized_rb:
	replay_buffer = PrioritizedReplayBuffer(size=memory_capacity, default_dtype=np.float32, env_dict=env_dict)
else:
	replay_buffer = ReplayBuffer(size=memory_capacity, default_dtype=np.float32, env_dict=env_dict)

# Train agent
incumbent_lap_float, incumbent_frame_count = None, None
evaluation_steps, evaluation_lap_float, evaluation_frame_count = [], [], []
obs = env.reset()
for total_steps in range(max_steps):

	# take and store action
	action = agent.get_action(obs, test=False)
	next_obs, reward, done, _ = env.step(action)
	replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done)
	obs = next_obs
	if done:
		obs = env.reset()

	# extract samples from buffer and train
	samples = replay_buffer.sample(batch_size)
	agent.train(
		samples["obs"], samples["act"], samples["next_obs"],
		samples["rew"], np.array(samples["done"], dtype=np.float32),
		None if not use_prioritized_rb else samples["weights"])
	if use_prioritized_rb:
		td_error = agent.compute_td_error(
			samples["obs"], samples["act"], samples["next_obs"],
			samples["rew"], np.array(samples["done"], dtype=np.float32))
		replay_buffer.update_priorities(samples["indexes"], np.abs(td_error) + 1e-6)

	# evaluate performance
	if (total_steps == 0) or (total_steps % test_interval == 0):
		episode_lap_float, episode_frame_count, _, _ = evaluate_policy(False)
		evaluation_lap_float.append(episode_lap_float)
		evaluation_frame_count.append(episode_frame_count)
		evaluation_steps.append(total_steps)
		date_str = f"{datetime.now():%H-%M-%S}"
		filename = f"{date_str}_{episode_lap_float:.2f}_{episode_frame_count:3}"
		print_str = f"{date_str}: Steps={total_steps: 7}, Laps={episode_lap_float:.2f}, Frames={episode_frame_count:3}"
		save_best = False
		if incumbent_lap_float == None or incumbent_frame_count == None:
			save_best = True
		elif episode_lap_float > incumbent_lap_float or (episode_lap_float == incumbent_lap_float and episode_frame_count < incumbent_frame_count):
			save_best = True
		if save_best:
			_, _, episode_render, episode_metrics = evaluate_policy(True)
			print(print_str + " saved")
			incumbent_lap_float, incumbent_frame_count = episode_lap_float, episode_frame_count
			agent.save_agent(filename)
			if len(episode_render) > 0:
				display_episode(episode_render, episode_metrics, f"{agent.alg_name}_{filename}")
		else:
			print(print_str)

# Visualise results
print("All episodes done")
fig2, ax2_1 = plt.subplots()
ax2_1.set_xlabel("Steps")
color = 'tab:red'
ax2_1.plot(evaluation_steps, [x*env.lap_length/(y*env.time_per_frame) for x,y in zip(evaluation_lap_float,evaluation_frame_count)] , color=color)
ax2_1.set_ylabel("Speed [m/s]", color=color)
ax2_1.tick_params(axis='y', labelcolor=color)
ax2_2 = ax2_1.twinx()
color = 'tab:blue'
ax2_2.plot(evaluation_steps, evaluation_lap_float, color=color)
ax2_2.set_ylabel(f"Laps", color=color)
ax2_2.tick_params(axis='y', labelcolor=color)
fig2.tight_layout()
plt.show()