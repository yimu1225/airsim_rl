import argparse


def get_config(argv=None):
    # get the parameters
    parser = argparse.ArgumentParser(description='AirSim_RL')

    # env
    parser.add_argument("--env_name", type=str, default='AirSimEnv-v42')  # AirSimEnv-v42  CartPole-v0

    # prepare
    parser.add_argument("--algorithm_name", type=str, default='gru_td3', 
                        help="Algorithm to train. Can be single ('td3'), multiple ('td3,gru_td3'), or group ('all', 'base', 'seq')")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action='store_false', default=True)
    parser.add_argument("--cuda_deterministic", action='store_false', default=True)
    parser.add_argument("--n_training_threads", type=int, default=2)
    parser.add_argument("--n_rollout_threads", type=int, default=2)#this must be 1 in airsim env
    parser.add_argument("--num_env_steps", type=int, default=3e5, help='number of environment steps to train (default: 10e6)')

    # lstm
    parser.add_argument("--recurrent_policy", action='store_false', default=False, help='use a recurrent policy')
    parser.add_argument("--data_chunk_length", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--recurrent_input_size", type=int, default=512)# the feature dims of visual extractor output
    parser.add_argument("--recurrent_hidden_size", type=int, default=512)

    # ppo
    parser.add_argument("--ppo_epoch", type=int, default=8, help='number of ppo epochs (default: 4)')
    parser.add_argument("--use_clipped_value_loss", action='store_false', default=True)
    parser.add_argument("--clip_param", type=float, default=0.15, help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1, help='number of batches for ppo (default: 32)')
    parser.add_argument("--entropy_coef", type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float, default=1.0, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--lr", type=float, default=5e-4, help='learning rate (default: 7e-4)')
    parser.add_argument("--eps", type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--max-grad-norm", type=float, default=5, help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use-gae", action='store_false', default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae-lambda", type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use-proper-time-limits", action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=False)
    parser.add_argument("--huber_delta", type=float, default=10.0)


    # replay buffer
    parser.add_argument("--episode_length", type=int, default=512, help='number of forward steps in A2C (default: 5)')
    parser.add_argument("--steps_per_update", type=int, default=100, help='number of steps to collect before each update')


    # run
    parser.add_argument("--use-linear-lr-decay", action='store_false', default=False, help='use a linear schedule on the learning rate')
    
    # save
    parser.add_argument("--save_interval", type=int, default=20, help="Checkpoint frequency (in epochs)")
    parser.add_argument("--continue_last", default=False)

    # log
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=50, help="Number of steps per worker per epoch (Tune iteration)")

    #eval
    parser.add_argument("--eval", action='store_true', default=False)
    parser.add_argument("--save_gifs", action='store_true', default=False)
    parser.add_argument("--ifi", type=float, default=0.333333)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--model_dir", type=str, default='results/AirSimEnv-v42/ppo-lstm3/run5/models')

    # TD3
    parser.add_argument("--max_timesteps", type=int, default=1000000)
    parser.add_argument("--start_timesteps", type=int, default=2000)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--depth_feature_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--exploration_noise", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    # parser.add_argument("--gamma", type=float, default=0.99) # Already exist in PPO section
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    
    # OU Noise for TD3
    parser.add_argument("--ou_theta", type=float, default=0.15)
    parser.add_argument("--ou_sigma", type=float, default=0.2)
    parser.add_argument("--ou_sigma_min", type=float, default=0.01)
    parser.add_argument("--ou_dt", type=float, default=1.0)

    # GRU-TD3
    parser.add_argument("--seq_len", type=int, default=8, help="Sequence length for GRU")
    parser.add_argument("--gru_hidden_dim", type=int, default=256, help="Hidden dimension for GRU")
    parser.add_argument("--gru_num_layers", type=int, default=1, help="Number of GRU layers")
    
    # Continuous Control Parameters
    parser.add_argument("--min_forward_speed", type=float, default=0.0, help="Minimum forward speed (m/s)")
    parser.add_argument("--max_forward_speed", type=float, default=2.0, help="Maximum forward speed (m/s)")
    parser.add_argument("--max_vertical_speed", type=float, default=0.3, help="Maximum vertical speed (m/s)")
    parser.add_argument("--max_yaw_rate", type=float, default=1.0, help="Maximum yaw rate (rad/s)")
    parser.add_argument("--takeoff_height", type=float, default=-2.0, help="Target height for takeoff (negative is up in NED)")
    parser.add_argument("--action_duration", type=float, default=1.0,
                        help="Base duration (seconds) for each velocity command before clock scaling")
    parser.add_argument("--clock_speed_factor", type=float, default=5.0,
                        help="ClockSpeed factor configured in AirSim settings; durations are divided by this value")
    
    # Flight altitude limits
    parser.add_argument("--max_flight_altitude", type=float, default=4.0, help="Maximum flight altitude (m, positive is up)")
    parser.add_argument("--min_flight_altitude", type=float, default=0.5, help="Minimum flight altitude (m, positive is up)")
    parser.add_argument("--min_altitude_penalty", type=float, default=0.5, help="Penalty for flying below minimum altitude")    
    # AirSim connection parameters
    parser.add_argument("--airsim_ip", type=str, default="172.20.176.1", help="AirSim server IP address")
    parser.add_argument("--airsim_port", type=int, default=41451, help="AirSim server port")
    parser.add_argument("--settings_file", type=str, default="", help="AirSim settings.json file path (optional)")    
    parser.add_argument("--load_model", type=str, default="")

    # Ray Tune workers inject additional CLI args; ignore unknowns for compatibility.
    args, _ = parser.parse_known_args(args=argv)

    return args
