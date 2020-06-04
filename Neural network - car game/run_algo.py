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
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import tensorflow as tf

from algos import DQN, DDPG
from homegym import CarGameEnv

# Define user functions
def evaluate_policy():
    render_frames=[]
    metrics_dict = {"laps": [], "speed": [], "grip": [], "longitudinal_force": [], "steering_angle": []}
    obs = test_env.reset()
    done = False
    while not done:
        action = agent.get_action(obs, test=True)
        next_obs, _, done, (laps, speed, grip, longitudinal_force, steering_angle) = test_env.step(action)
        if show_test_progress:
            render_frames.append(test_env.render())
            metrics_dict["laps"].append(laps)
            metrics_dict["speed"].append(speed)
            metrics_dict["grip"].append(grip)
            metrics_dict["longitudinal_force"].append(longitudinal_force)
            metrics_dict["steering_angle"].append(steering_angle)
        obs = next_obs
    return test_env.lap_float, test_env.frame_curr, render_frames, metrics_dict

def display_episode(render, metrics, filedir):
	# initialise figure
    fig1, (ax1_1, ax1_2_1) = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [4,1]}, figsize=(7,6))
    fig1.tight_layout()
    # plot metrics
    ax1_2_1.plot(metrics["laps"],metrics["speed"], label="speed", color="tab:blue")
    ax1_2_1.plot(metrics["laps"],[x/1000 for x in metrics["grip"]], label="grip", color="tab:orange")
    ax1_2_2 = ax1_2_1.twinx()
    ax1_2_2.plot(metrics["laps"],[x/1000 for x in metrics["longitudinal_force"]], label="long_force", color="tab:green")
    ax1_2_2.plot(metrics["laps"],[x*180/pi for x in metrics["steering_angle"]], label="steer_ang", color="tab:red")
    h_1, l_1 = ax1_2_1.get_legend_handles_labels()
    h_2, l_2 = ax1_2_2.get_legend_handles_labels()
    ax1_2_2.legend(h_1+h_2, l_1+l_2, bbox_to_anchor=(1.05, 1.0), loc="upper left")
    #initialise image and metric markers
    img_1 = ax1_1.imshow(render[0])
    ax1_1.axis('off')
    l_speed, = ax1_2_1.plot(metrics["laps"][0],metrics["speed"][0], "o", color="tab:blue")
    l_grip, = ax1_2_1.plot(metrics["laps"][0],metrics["grip"][0]/1000, "o", color="tab:orange")
    l_longitudinal_force, = ax1_2_2.plot(metrics["laps"][0],metrics["longitudinal_force"][0]/1000, "o", color="tab:green")
    l_steering_angle, = ax1_2_2.plot(metrics["laps"][0],metrics["steering_angle"][0]*180/pi, "o", color="tab:red")

    def update(frame):
        img_1.set_data(render[frame])
        l_speed.set_data(metrics["laps"][frame],metrics["speed"][frame])
        l_grip.set_data(metrics["laps"][frame],metrics["grip"][frame]/1000)
        l_longitudinal_force.set_data(metrics["laps"][frame],metrics["longitudinal_force"][frame]/1000)
        l_steering_angle.set_data(metrics["laps"][frame],metrics["steering_angle"][frame]*180/pi)
        return img_1, l_speed, l_grip, l_longitudinal_force, l_steering_angle

    # create and save animation
    ani = FuncAnimation(fig1, update, frames=range(len(render)), blit=True, interval=33)
    ani.save(filedir + "/annotated.gif", dpi=150)

    # save video without metrics
    with imageio.get_writer(filedir + "/raw.mp4", fps=30) as video:
        for frame in render:
            video.append_data(frame)

# Define variables
continuous = True
use_prioritized_rb=True
show_test_progress=True
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
        load_model="DDPG_23-03-59_2.00_281",
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
        episode_lap_float, episode_frame_count, episode_render, episode_metrics = evaluate_policy()
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
ax2_1.set_xlabel("Episode")
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