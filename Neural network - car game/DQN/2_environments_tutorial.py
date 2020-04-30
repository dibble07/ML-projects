# ## Introduction

# The goal of Reinforcement Learning (RL) is to design agents that learn by interacting with an environment. In the standard RL setting, the agent receives an observation at every time step and chooses an action. The action is applied to the environment and the environment returns a reward and a new observation. The agent trains a policy to choose actions to maximize the sum of rewards, also known as return.
# 
# In TF-Agents, environments can be implemented either in Python or TensorFlow. Python environments are usually easier to implement, understand, and debug, but TensorFlow environments are more efficient and allow natural parallelization. The most common workflow is to implement an environment in Python and use one of our wrappers to automatically convert it into TensorFlow.
# 
# Let us look at Python environments first. TensorFlow environments follow a very similar API.

# ## Setup
# 

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

tf.compat.v1.enable_v2_behavior()

# ## Python Environments

# Python environments have a `step(action) -> next_time_step` method that applies an action to the environment, and returns the following information about the next step:
# 1. `observation`: This is the part of the environment state that the agent can observe to choose its actions at the next step.
# 2. `reward`: The agent is learning to maximize the sum of these rewards across multiple steps.
# 3. `step_type`: Interactions with the environment are usually part of a sequence/episode. e.g. multiple moves in a game of chess. step_type can be either `FIRST`, `MID` or `LAST` to indicate whether this time step is the first, intermediate or last step in a sequence.
# 4. `discount`: This is a float representing how much to weight the reward at the next time step relative to the reward at the current time step.
# 
# These are grouped into a named tuple `TimeStep(step_type, reward, discount, observation)`.
# 
# The interface that all python environments must implement is in `environments/py_environment.PyEnvironment`. The main methods are:

# In[ ]:


class PyEnvironment(object):

  def reset(self):
    """Return initial_time_step."""
    self._current_time_step = self._reset()
    return self._current_time_step

  def step(self, action):
    """Apply action and return new time_step."""
    if self._current_time_step is None:
        return self.reset()
    self._current_time_step = self._step(action)
    return self._current_time_step

  def current_time_step(self):
    return self._current_time_step

  def time_step_spec(self):
    """Return time_step_spec."""

  @abc.abstractmethod
  def observation_spec(self):
    """Return observation_spec."""

  @abc.abstractmethod
  def action_spec(self):
    """Return action_spec."""

  @abc.abstractmethod
  def _reset(self):
    """Return initial_time_step."""

  @abc.abstractmethod
  def _step(self, action):
    """Apply action and return new time_step."""
    self._current_time_step = self._step(action)
    return self._current_time_step

# In addition to the `step()` method, environments also provide a `reset()` method that starts a new sequence and provides an initial `TimeStep`. It is not necessary to call the `reset` method explicitly. We assume that environments reset automatically, either when they get to the end of an episode or when step() is called the first time.
# 
# Note that subclasses do not implement `step()` or `reset()` directly. They instead override the `_step()` and `_reset()` methods. The time steps returned from  these methods will be cached and exposed through `current_time_step()`.
# 
# The `observation_spec` and the `action_spec` methods return a nest of `(Bounded)ArraySpecs` that describe the name, shape, datatype and ranges of the observations and actions respectively.
# 
# In TF-Agents we repeatedly refer to nests which are defined as any tree like structure composed of lists, tuples, named-tuples, or dictionaries. These can be arbitrarily composed to maintain structure of observations and actions. We have found this to be very useful for more complex environments where you have many observations and actions.

# ### Using Standard Environments
# 
# TF Agents has built-in wrappers for many standard environments like the OpenAI Gym, DeepMind-control and Atari, so that they follow our `py_environment.PyEnvironment` interface. These wrapped evironments can be easily loaded using our environment suites. Let's load the CartPole environment from the OpenAI gym and look at the action and time_step_spec.

# In[ ]:


environment = suite_gym.load('CartPole-v0')
print('action_spec:', environment.action_spec())
print('time_step_spec.observation:', environment.time_step_spec().observation)
print('time_step_spec.step_type:', environment.time_step_spec().step_type)
print('time_step_spec.discount:', environment.time_step_spec().discount)
print('time_step_spec.reward:', environment.time_step_spec().reward)


# So we see that the environment expects actions of type `int64` in [0, 1] and returns `TimeSteps` where the observations are a `float32` vector of length 4 and discount factor is a `float32` in [0.0, 1.0]. Now, let's try to take a fixed action `(1,)` for a whole episode.

# In[ ]:


action = np.array(1, dtype=np.int32)
time_step = environment.reset()
print(time_step)
while not time_step.is_last():
  time_step = environment.step(action)
  print(time_step)

# ### Creating your own Python Environment
# 
# For many clients, a common use case is to apply one of the standard agents (see agents/) in TF-Agents to their problem. To do this, they have to frame their problem as an environment. So let us look at how to implement an environment in Python.
# 
# Let's say we want to train an agent to play the following (Black Jack inspired) card game:
# 
# 1. The game is played using an infinite deck of cards numbered 1...10.
# 2. At every turn the agent can do 2 things: get a new random card, or stop the current round.
# 3. The goal is to get the sum of your cards as close to 21 as possible at the end of the round, without going over.
# 
# An environment that represents the game could look like this:
# 
# 1. Actions: We have 2 actions. Action 0: get a new card, and Action 1: terminate the current round.
# 2. Observations: Sum of the cards in the current round.
# 3. Reward: The objective is to get as close to 21 as possible without going over, so we can achieve this using the following reward at the end of the round:
#    sum_of_cards - 21 if sum_of_cards <= 21, else -21
# 

# In[ ]:


class CardGameEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.int32, minimum=0, name='observation')
    self._state = 0
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = 0
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # Make sure episodes don't go on forever.
    if action == 1:
      self._episode_ended = True
    elif action == 0:
      new_card = np.random.randint(1, 11)
      self._state += new_card
    else:
      raise ValueError('`action` should be 0 or 1.')

    if self._episode_ended or self._state >= 21:
      reward = self._state - 21 if self._state <= 21 else -21
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      return ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

# Let's make sure we did everything correctly defining the above environment. When creating your own environment you must make sure the observations and time_steps generated follow the correct shapes and types as defined in your specs. These are used to generate the TensorFlow graph and as such can create hard to debug problems if we get them wrong.
# 
# To validate our environment we will use a random policy to generate actions and we will iterate over 5 episodes to make sure things are working as intended. An error is raised if we receive a time_step that does not follow the environment specs.

# In[ ]:


environment = CardGameEnv()
utils.validate_py_environment(environment, episodes=5)

# Now that we know the environment is working as intended, let's run this environment using a fixed policy: ask for 3 cards and then end the round.

# In[ ]:


get_new_card_action = np.array(0, dtype=np.int32)
end_round_action = np.array(1, dtype=np.int32)

environment = CardGameEnv()
time_step = environment.reset()
print(time_step)
cumulative_reward = time_step.reward

for _ in range(3):
  time_step = environment.step(get_new_card_action)
  print(time_step)
  cumulative_reward += time_step.reward

time_step = environment.step(end_round_action)
print(time_step)
cumulative_reward += time_step.reward
print('Final Reward = ', cumulative_reward)

# ### Environment Wrappers
# 
# An environment wrapper takes a python environment and returns a modified version of the environment. Both the original environment and the modified environment are instances of `py_environment.PyEnvironment`, and multiple wrappers can be chained together.
# 
# Some common wrappers can be found in `environments/wrappers.py`. For example:
# 
# 1. `ActionDiscretizeWrapper`: Converts a continuous action space to a discrete action space.
# 2. `RunStats`: Captures run statistics of the environment such as number of steps taken, number of episodes completed etc.
# 3. `TimeLimit`: Terminates the episode after a fixed number of steps.
# 

# #### Example 1: Action Discretize Wrapper

# InvertedPendulum is a PyBullet environment that accepts continuous actions in the range `[-2, 2]`. If we want to train a discrete action agent such as DQN on this environment, we have to discretize (quantize) the action space. This is exactly what the `ActionDiscretizeWrapper` does. Compare the `action_spec` before and after wrapping:

# In[ ]:


env = suite_gym.load('Pendulum-v0')
print('Action Spec:', env.action_spec())

discrete_action_env = wrappers.ActionDiscretizeWrapper(env, num_actions=5)
print('Discretized Action Spec:', discrete_action_env.action_spec())

# The wrapped `discrete_action_env` is an instance of `py_environment.PyEnvironment` and can be treated like a regular python environment.
# 

# ## TensorFlow Environments

# The interface for TF environments is defined in `environments/tf_environment.TFEnvironment` and looks very similar to the Python environments. TF Environments differ from python envs in a couple of ways:
# 
# * They generate tensor objects instead of arrays
# * TF environments add a batch dimension to the tensors generated when compared to the specs. 
# 
# Converting the python environments into TFEnvs allows tensorflow to parellalize operations. For example, one could define a `collect_experience_op` that collects data from the environment and adds to a `replay_buffer`, and a `train_op` that reads from the `replay_buffer` and trains the agent, and run them in parallel naturally in TensorFlow.

# In[ ]:


class TFEnvironment(object):

  def time_step_spec(self):
    """Describes the `TimeStep` tensors returned by `step()`."""

  def observation_spec(self):
    """Defines the `TensorSpec` of observations provided by the environment."""

  def action_spec(self):
    """Describes the TensorSpecs of the action expected by `step(action)`."""

  def reset(self):
    """Returns the current `TimeStep` after resetting the Environment."""
    return self._reset()

  def current_time_step(self):
    """Returns the current `TimeStep`."""
    return self._current_time_step()

  def step(self, action):
    """Applies the action and returns the new `TimeStep`."""
    return self._step(action)

  @abc.abstractmethod
  def _reset(self):
    """Returns the current `TimeStep` after resetting the Environment."""

  @abc.abstractmethod
  def _current_time_step(self):
    """Returns the current `TimeStep`."""

  @abc.abstractmethod
  def _step(self, action):
    """Applies the action and returns the new `TimeStep`."""

# The `current_time_step()` method returns the current time_step and initializes the environment if needed.
# 
# The `reset()` method forces a reset in the environment and returns the current_step.
# 
# If the `action` doesn't depend on the previous `time_step` a `tf.control_dependency` is needed in `Graph` mode.
# 
# For now, let us look at how `TFEnvironments` are created.

# ### Creating your own TensorFlow Environment
# 
# This is more complicated than creating environments in Python, so we will not cover it in this colab. An example is available [here](https://github.com/tensorflow/agents/blob/master/tf_agents/environments/tf_environment_test.py). The more common use case is to implement your environment in Python and wrap it in TensorFlow using our `TFPyEnvironment` wrapper (see below).

# ### Wrapping a Python Environment in TensorFlow

# We can easily wrap any Python environment into a TensorFlow environment using the `TFPyEnvironment` wrapper.

# In[ ]:


env = suite_gym.load('CartPole-v0')
tf_env = tf_py_environment.TFPyEnvironment(env)

print(isinstance(tf_env, tf_environment.TFEnvironment))
print("TimeStep Specs:", tf_env.time_step_spec())
print("Action Specs:", tf_env.action_spec())

# Note the specs are now of type: `(Bounded)TensorSpec`.

# ### Usage Examples

# #### Simple Example

# In[ ]:


env = suite_gym.load('CartPole-v0')

tf_env = tf_py_environment.TFPyEnvironment(env)
# reset() creates the initial time_step after resetting the environment.
time_step = tf_env.reset()
num_steps = 3
transitions = []
reward = 0
for i in range(num_steps):
  action = tf.constant([i % 2])
  # applies the action and returns the new TimeStep.
  next_time_step = tf_env.step(action)
  transitions.append([time_step, action, next_time_step])
  reward += next_time_step.reward
  time_step = next_time_step

np_transitions = tf.nest.map_structure(lambda x: x.numpy(), transitions)
print('\n'.join(map(str, np_transitions)))
print('Total reward:', reward.numpy())

# #### Whole Episodes

# In[ ]:


env = suite_gym.load('CartPole-v0')
tf_env = tf_py_environment.TFPyEnvironment(env)

time_step = tf_env.reset()
rewards = []
steps = []
num_episodes = 5

for _ in range(num_episodes):
  episode_reward = 0
  episode_steps = 0
  while not time_step.is_last():
    action = tf.random.uniform([1], 0, 2, dtype=tf.int32)
    time_step = tf_env.step(action)
    episode_steps += 1
    episode_reward += time_step.reward.numpy()
  rewards.append(episode_reward)
  steps.append(episode_steps)
  time_step = tf_env.reset()

num_steps = np.sum(steps)
avg_length = np.mean(steps)
avg_reward = np.mean(rewards)

print('num_episodes:', num_episodes, 'num_steps:', num_steps)
print('avg_length', avg_length, 'avg_reward:', avg_reward)
