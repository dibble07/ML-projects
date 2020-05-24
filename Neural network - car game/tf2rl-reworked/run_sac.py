import gym

from tf2rl.algos.sac import SAC
from tf2rl.experiments.trainer import Trainer

from homegym import CarGameEnv, LunarLanderContinuous

if __name__ == '__main__':
    parser = Trainer.get_argument()
    print(parser.parse_args())
    parser = SAC.get_argument(parser)
    print(parser.parse_args())
    # parser.add_argument('--env-name', type=str, default="Pendulum-v0")
    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=10000)
    parser.set_defaults(max_steps=1e6)
    args = parser.parse_args()
    print(parser.parse_args())

    # env = gym.make(args.env_name)
    # test_env = gym.make(args.env_name)
    env = LunarLanderContinuous()
    test_env = LunarLanderContinuous()
    env = CarGameEnv(True)
    test_env = CarGameEnv(True)
    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        # gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup,
        alpha=args.alpha)#, auto_alpha=args.auto_alpha
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
