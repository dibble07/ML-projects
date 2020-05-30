# Import libraries
import os

import numpy as np
import tensorflow as tf
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from cpprb import ReplayBuffer, PrioritizedReplayBuffer

from algos import DQN, DDPG
from homegym import CarGameEnv

# Define user functions
def evaluate_policy():
    episode_return = 0.
    obs = test_env.reset()
    done = False
    while not done:
        action = agent.get_action(obs, test=True)
        next_obs, reward, done, _ = test_env.step(action)
        if show_test_progress:
            test_env.render()
        episode_return += reward
        obs = next_obs
    return episode_return

# Define variables
continuous = False
use_prioritized_rb=False
show_test_progress=True
max_steps=1_000_000
test_interval=1_000
memory_capacity=1_000_000
batch_size=64

# Initialise environment, policy and replay buffer
env = CarGameEnv(continuous)
test_env = CarGameEnv(continuous)
if continuous:
    agent = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        discount=0.99,
        max_action=env.action_space.high[0],
        load_model="best"
        )
else:
    agent = DQN(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        discount=0.99,
        enable_double_dqn=True,
        enable_dueling_dqn=True,
        target_replace_interval=300,
        load_model="best"
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
best_score = None
obs = env.reset()
for total_steps in range(max_steps):

    # take and store action
    action = agent.get_action(obs)
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
        avg_test_return = evaluate_policy()
        save_flag=False
        print("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f}".format(total_steps, avg_test_return))
        if best_score is None or avg_test_return>best_score:
            print(f"{best_score:.3f}, {avg_test_return:.3f}")
            agent.save_agent()
            best_score=avg_test_return
