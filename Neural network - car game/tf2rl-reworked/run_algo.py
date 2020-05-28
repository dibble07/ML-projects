import gym

from tf2rl.algos.dqn import DQN
from tf2rl.algos.ddpg import DDPG
from tf2rl.experiments.trainer import Trainer

from homegym import CarGameEnv

continuous = True

env = CarGameEnv(continuous)
test_env = CarGameEnv(continuous)

if continuous:
    policy = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        discount=0.99,
        max_action=env.action_space.high[0])
else:
    policy = DQN(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        discount=0.99,
        enable_double_dqn=False,
        enable_dueling_dqn=False,
        enable_noisy_dqn=False,
        target_replace_interval=300)

trainer = Trainer(
    policy=policy,
    env=env,
    use_prioritized_rb=False,
    show_test_progress=True,
    max_steps=1e6,
    test_interval=1e3,
    memory_capacity=1e6,
    batch_size=64)
trainer()