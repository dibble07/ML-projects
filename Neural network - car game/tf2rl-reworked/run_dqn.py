import gym

from tf2rl.algos.dqn import DQN
from tf2rl.experiments.trainer import Trainer

from homegym import CarGameEnv

if __name__ == '__main__':

    env = CarGameEnv(False)
    test_env = CarGameEnv(False)

    policy = DQN(
        enable_double_dqn=False,
        enable_dueling_dqn=False,
        enable_noisy_dqn=False,
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        target_replace_interval=300,
        discount=0.99,
        gpu=0,
        memory_capacity=1e6,
        batch_size=64)

    trainer = Trainer(policy, env, use_prioritized_rb=False, show_test_progress=True, max_steps=1e6, test_interval=1e3)
    trainer()