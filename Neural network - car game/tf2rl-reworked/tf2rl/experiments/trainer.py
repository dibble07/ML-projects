import os

import numpy as np
import tensorflow as tf

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from cpprb import ReplayBuffer, PrioritizedReplayBuffer

def get_space_size(space):
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return [1, ]  # space.n
    else:
        raise NotImplementedError("Assuming to use Box or Discrete, not {}".format(type(space)))


def get_default_rb_dict(env):
    return {
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {
                "shape": get_space_size(env.observation_space)},
            "next_obs": {
                "shape": get_space_size(env.observation_space)},
            "act": {
                "shape": get_space_size(env.action_space)},
            "rew": {},
            "done": {}}}


def get_replay_buffer(env, use_prioritized_rb, size):
    obs_shape = get_space_size(env.observation_space)
    kwargs = get_default_rb_dict(env)
    kwargs["size"] = size

    if use_prioritized_rb:
        return PrioritizedReplayBuffer(**kwargs)
    else:
        return ReplayBuffer(**kwargs)


class Trainer:
    def __init__(self, policy, env, use_prioritized_rb, show_test_progress, max_steps, test_interval, memory_capacity, batch_size):
        self._policy = policy
        self._env = env
        self._test_env = self._env
        self._use_prioritized_rb = use_prioritized_rb
        self._show_test_progress = show_test_progress
        self._max_steps = max_steps
        self._test_interval = test_interval
        self._memory_capacity = memory_capacity
        self._batch_size = batch_size

    def __call__(self):
        total_steps = 0
        episode_steps = 0
        episode_return = 0
        n_episode = 0

        replay_buffer = get_replay_buffer(self._env, self._use_prioritized_rb, self._memory_capacity)

        obs = self._env.reset()

        while total_steps < self._max_steps:
            action = self._policy.get_action(obs)

            next_obs, reward, done, _ = self._env.step(action)
            episode_steps += 1
            episode_return += reward
            total_steps += 1

            done_flag = done
            if hasattr(self._env, "_max_episode_steps") and episode_steps == self._env._max_episode_steps:
                done_flag = False
            replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done_flag)
            obs = next_obs

            if done:
                obs = self._env.reset()

                n_episode += 1
                episode_steps = 0
                episode_return = 0

            # if total_steps % self._policy.update_interval == 0:
            samples = replay_buffer.sample(self._batch_size)
            self._policy.train(
                samples["obs"], samples["act"], samples["next_obs"],
                samples["rew"], np.array(samples["done"], dtype=np.float32),
                None if not self._use_prioritized_rb else samples["weights"])
            if self._use_prioritized_rb:
                td_error = self._policy.compute_td_error(
                    samples["obs"], samples["act"], samples["next_obs"],
                    samples["rew"], np.array(samples["done"], dtype=np.float32))
                replay_buffer.update_priorities(samples["indexes"], np.abs(td_error) + 1e-6)

            if total_steps % self._test_interval == 0:
                avg_test_return = self.evaluate_policy(total_steps)
                print("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f}".format(total_steps, avg_test_return))

    def evaluate_policy(self, total_steps):
        episode_return = 0.
        obs = self._test_env.reset()
        done = False
        while not done:
            action = self._policy.get_action(obs, test=True)
            next_obs, reward, done, _ = self._test_env.step(action)
            if self._show_test_progress:
                self._test_env.render()
            episode_return += reward
            obs = next_obs
        return episode_return