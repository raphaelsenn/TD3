from argparse import Namespace, ArgumentParser

import torch
import gymnasium as gym

from td3.actor import ActorMLP


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Online Evaluation")

    parser.add_argument("--env_id", type=str, default="HalfCheetah-v5")
    parser.add_argument("--state_dim", type=int, default=17)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--h1_dim", type=int, default=400)
    parser.add_argument("--h2_dim", type=int, default=300)
    parser.add_argument("--weights", type=str, default="HalfCheetah-v5-TD3-Checkpoints/HalfCheetah-v5-TD3-Actor-Lr0.0003-t980000.pt")

    parser.add_argument("--verbose", default=True)

    return parser.parse_args()


def play(env: gym.Env, actor: ActorMLP, n_episodes: int=10) -> None:
    for ep in range(n_episodes):
        done = False
        s, _ = env.reset()
        while not done:
            a = actor.predict(s)
            s_nxt, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated 
            s = s_nxt
    env.close()


def main() -> None:
    args = parse_args()
    
    actor = ActorMLP(args.state_dim, args.h1_dim, args.h2_dim, args.action_dim)
    actor.load_state_dict(torch.load(args.weights, weights_only=True, map_location="cpu")) 
    env = gym.make(args.env_id, render_mode="human")
    play(env, actor)


if __name__ == "__main__":
    main()