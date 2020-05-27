import os
# import time
# import argparse

import numpy as np
import tensorflow as tf
# from gym.spaces import Box

from tf2rl.misc.get_replay_buffer import get_replay_buffer

class Trainer:
    def __init__(self, policy, env, use_prioritized_rb, show_test_progress, max_steps, test_interval):
        self._policy = policy
        self._env = env
        self._test_env = self._env
        self._use_prioritized_rb = use_prioritized_rb
        self._show_test_progress = show_test_progress
        self._max_steps = max_steps
        self._test_interval = test_interval

    def __call__(self):
        total_steps = 0
        episode_steps = 0
        episode_return = 0
        # episode_start_time = time.perf_counter()
        n_episode = 0

        replay_buffer = get_replay_buffer(
            self._policy, self._env, self._use_prioritized_rb)

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
                # fps = episode_steps / (time.perf_counter() - episode_start_time)
                episode_steps = 0
                episode_return = 0
                # episode_start_time = time.perf_counter()

            if total_steps % self._policy.update_interval == 0:
                samples = replay_buffer.sample(self._policy.batch_size)
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