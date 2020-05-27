import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from cpprb import ReplayBuffer, PrioritizedReplayBuffer

from tf2rl.algos.policy_base import OffPolicyAgent

def get_space_size(space):
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return [1, ]  # space.n
    else:
        raise NotImplementedError("Assuming to use Box or Discrete, not {}".format(type(space)))


def get_default_rb_dict(size, env):
    return {
        "size": size,
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


def get_replay_buffer(policy, env, use_prioritized_rb=False, size=None):
    obs_shape = get_space_size(env.observation_space)
    kwargs = get_default_rb_dict(policy.memory_capacity, env)

    if size is not None:
        kwargs["size"] = size

    if len(obs_shape) == 3:
        kwargs["env_dict"]["obs"]["dtype"] = np.ubyte
        kwargs["env_dict"]["next_obs"]["dtype"] = np.ubyte

    # prioritized
    if use_prioritized_rb:
        return PrioritizedReplayBuffer(**kwargs)

    return ReplayBuffer(**kwargs)
