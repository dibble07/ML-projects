import gym

from tf2rl.algos.ddpg import DDPG
from tf2rl.experiments.trainer import Trainer

from homegym import CarGameEnv

if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = DDPG.get_argument(parser)
    parser.set_defaults(batch_size=100)
    parser.set_defaults(show_test_progress=True)
    args = parser.parse_args()

    env = CarGameEnv(True)
    test_env = CarGameEnv(True)
    policy = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
