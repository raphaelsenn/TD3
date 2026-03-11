from argparse import Namespace, ArgumentParser

from td3.td3 import TD3
from td3.actor import ActorMLP
from td3.critic import CriticMLP

import torch
import gymnasium as gym
import numpy as np


def parse_args() -> Namespace:
    parser = ArgumentParser(description="TD3 training")

    parser.add_argument("--env_id", type=str, default="HalfCheetah-v5")
    parser.add_argument("--num_timesteps", type=int, default=1_000_000)
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--h1_dim", type=int, default=256)
    parser.add_argument("--h2_dim", type=int, default=256)

    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--weight_decay_actor", type=float, default=0.0)
    parser.add_argument("--weight_decay_critic", type=float, default=0.0)
    parser.add_argument("--delay", type=int, default=2)

    parser.add_argument("--buffer_capacity", type=int, default=1_000_000)
    parser.add_argument("--buffer_start_size", type=int, default=25_000)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exp_noise", type=float, default=0.1)
    parser.add_argument("--tgt_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)

    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--eval_every", type=int, default=5_000)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", default=True)

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    
    env = gym.make(args.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = ActorMLP(state_dim, args.h1_dim, args.h2_dim, action_dim, args.action_scale)
    critic = CriticMLP(state_dim, args.h1_dim, args.h2_dim, action_dim)
    print(args)
    ddpg = TD3(
        actor=actor,
        critic=critic,
        timesteps=args.num_timesteps,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        weight_decay_actor=args.weight_decay_actor,
        weight_decay_critic=args.weight_decay_critic,
        delay=args.delay,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        gamma=args.gamma,
        tau=args.tau,
        exp_noise_std=args.exp_noise,
        tgt_noise_std=args.tgt_noise,
        noise_clip=args.noise_clip,
        device=args.device,
        buffer_start_size=args.buffer_start_size,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed

    ) 
    ddpg.train(env)


if __name__ == "__main__":
    main()