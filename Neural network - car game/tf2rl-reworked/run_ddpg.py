import gym

from tf2rl.algos.ddpg import DDPG
from tf2rl.experiments.trainer import Trainer

from homegym import CarGameEnv

if __name__ == '__main__':

    env = CarGameEnv(True)
    test_env = CarGameEnv(True)

    policy = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=0,
        memory_capacity=1e6,
        max_action=env.action_space.high[0],
        batch_size=64)
    
    trainer = Trainer(policy, env, use_prioritized_rb=False, show_test_progress=True, max_steps=1e6, test_interval=1e3)
    trainer()
