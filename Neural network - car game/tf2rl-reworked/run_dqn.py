import gym

from tf2rl.algos.dqn import DQN
from tf2rl.experiments.trainer import Trainer

from homegym import CarGameEnv

if __name__ == '__main__':
    parser = DQN.get_argument()
    parser.set_defaults(batch_size=64)
    args = parser.parse_args()
    print(parser.parse_args())

    env = CarGameEnv(False)
    test_env = CarGameEnv(False)
    policy = DQN(
        enable_double_dqn=args.enable_double_dqn,
        enable_dueling_dqn=args.enable_dueling_dqn,
        enable_noisy_dqn=args.enable_noisy_dqn,
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        target_replace_interval=300,
        discount=0.99,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        batch_size=args.batch_size,
        n_warmup=args.n_warmup)
    trainer = Trainer(policy, env, use_prioritized_rb=False, show_test_progress=True, max_steps=1e6, test_interval=1e3)
    trainer()
