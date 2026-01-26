import numpy as np
import torch
import argparse
from gym_airsim.envs.AirGym import AirSimEnv
from config import get_config
from training.train import train

def main():
    args = get_config()
    
    # Initialize Environment
    print("Initialize AirSimEnv...")
    print(f"AirSim connection: {args.airsim_ip}:{args.airsim_port}")
    if args.settings_file:
        print(f"Settings file: {args.settings_file}")
    
    try:
        env = AirSimEnv(need_render=False, takeoff_height=args.takeoff_height, config=args) 
    except Exception as e:
        print(f"Failed to initialize AirSimEnv: {e}")
        return

    # Set seeds
    # env.seed(args.seed) # Gymnasium handles seeding in reset
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # State dim & Action dim
    obs, info = env.reset(seed=args.seed)
    depth_image, inform = obs
    
    base_dim = inform.shape[0]
    depth_shape = depth_image.shape
    action_space = env.action_space
    
    print(f"Observation shapes: Depth {depth_shape}, Info {base_dim}")
    print(f"Action space: {action_space}")
    print(f"Algorithm: {args.algorithm_name}")

    # Initialize Agent based on Algorithm
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    agent = None
    if args.algorithm_name == "td3":
        from algorithm.td3.td3 import TD3Agent
        agent = TD3Agent(base_dim, depth_shape, action_space, args, device=device)
    elif args.algorithm_name == "gru_td3":
        from algorithm.gru_td3.gru_td3 import GRUTD3Agent
        agent = GRUTD3Agent(base_dim, depth_shape, action_space, args, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm_name}")
    
    if args.load_model != "":
         print(f"Loading model: {args.load_model}")
         agent.load(args.load_model)
    
    # Start Training
    train(env, agent, args)

if __name__ == "__main__":
    main()
