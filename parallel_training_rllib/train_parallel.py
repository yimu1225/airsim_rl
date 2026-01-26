
import os
import sys
import time
import shutil
import numpy as np
import torch
import ray
import copy

# Add root directory to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config import get_config
from algorithm.td3.td3 import TD3Agent
from algorithm.gru_td3.gru_td3 import GRUTD3Agent

# Try to import AirSimEnv

from gym_airsim.envs.AirGym import AirSimEnv


# Get global arguments
args = get_config()

@ray.remote
class RolloutWorker:
    def __init__(self, worker_id, port, seed, global_args, device='cpu'):
        self.worker_id = worker_id
        self.args = copy.deepcopy(global_args)
        self.device = device
        
        # Override port and IP
        self.args.airsim_port = port
        if not self.args.airsim_ip: 
            self.args.airsim_ip = ""
            
        # ----------------------------------------------------------------------
        # 重要修改: 禁止 Worker 自动重启游戏
        # 并行训练时必须由外部脚本 (如 start_ue4_instances.py) 统一管理 UE4 实例
        # 否则 Worker 之间会互相杀死对方的编辑器进程
        # ----------------------------------------------------------------------
        self.args.disable_game_restart = True 

        print(f"[Worker {worker_id}] Connecting to AirSim on port {port}...")
        
        # Instantiate Environment
        try:
            self.env = AirSimEnv(need_render=False, takeoff_height=self.args.takeoff_height, config=self.args)
            self.env.seed(seed)
        except Exception as e:
            print(f"[Worker {worker_id}] Env init failed: {e}")
            raise e
        
        # Handle Reset
        # AirSimEnv returns observation directly (Old Gym style)
        # Observation is a list: [depth_stack (list), inform (numpy)]
        self.state, _ = self.env.reset()
        if isinstance(self.state, tuple) and len(self.state) == 2 and isinstance(self.state[1], dict):
             # Detected Gymnasium style (obs, info), though unlikely for this env
             self.state = self.state[0]
        
        # Initialize Agent
        self.agent = self._create_agent(self.args, device)
        
    def _create_agent(self, args, device):
        algo = getattr(args, 'algorithm_name', 'gru_td3')
        
        # Guess dimensions from state
        # self.state is likely [depth_stack, inform]
        if isinstance(self.state, (list, tuple)) and len(self.state) == 2:
            d = self.state[0]
            inform = self.state[1]
            if isinstance(d, list) and len(d) > 0:
                depth_shape = (len(d), *d[0].shape)
            elif hasattr(d, 'shape'):
                depth_shape = d.shape
            else:
                depth_shape = (4, 112, 112) # Fallback
            
            base_dim = inform.shape[0] if hasattr(inform, 'shape') else 8
        else:
            base_dim = 8
            depth_shape = (4, 112, 112)
        
        if algo == 'td3':
            return TD3Agent(base_dim, depth_shape, self.env.action_space, args, device)
        elif algo == 'gru_td3':
            return GRUTD3Agent(base_dim, depth_shape, self.env.action_space, args, device)
        else:
            print(f"Unknown algorithm {algo}, defaulting to GRU-TD3")
            return GRUTD3Agent(base_dim, depth_shape, self.env.action_space, args, device)

    def sample(self, num_steps):
        data = []
        for _ in range(num_steps):
            # unpack state
            if isinstance(self.state, (list, tuple)) and len(self.state) == 2:
                depth_stack_list, base_state = self.state
            else:
                print(f"Error: Invalid state structure: {type(self.state)}")
                break # break to avoid crash
            
            depth_stack = np.array(depth_stack_list) 
            
            # Prepare input for Agent (Sequence format)
            base_seq = np.tile(base_state, (len(depth_stack), 1)) 
            depth_seq = depth_stack # (4, H, W)
            
            # Note:
            # depth_seq is (K, H, W)
            # base_seq is (K, D)
            # select_action expects (K, ...) inputs and internally adds/unsqueeze Batch dim (1, K, ...)
            
            # CRITICAL FIX for "ValueError: not enough values to unpack (expected 5, got 4)"
            # Inside GRUTD3Agent.select_action:
            #   base = torch.as_tensor(base_seq).unsqueeze(0)  -> (1, K, D)
            #   depth = torch.as_tensor(depth_seq).unsqueeze(0) -> (1, K, H, W) NOT (1, K, C, H, W)
            
            # The error `B, K, _, H, W = depth.shape` at line 145 implies depth tensor is 4D (B, K, H, W).
            # But the unpacking expects 5 dimensions (B, K, C, H, W).
            # This means the visual encoder expects a Channel dimension.
            
            # So, depth_seq must have shape (K, C, H, W).
            # Current depth_stack is (Stack, H, W) which corresponds to (K, H, W).
            # We need to add the channel dimension C=1.
            
            depth_seq = np.expand_dims(depth_stack, axis=1) # (K, 1, H, W)
            
            action = self.agent.select_action(base_seq, depth_seq, noise=self.args.exploration_noise)
            
            # Step - Handle API variation
            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                next_state, reward, done, info = step_result
            else:
                # Unexpected
                next_state, reward, done, info = step_result[0], step_result[1], step_result[2], {}
            
            transition = (self.state, action, reward, next_state, float(done))
            data.append(transition)
            
            self.state = next_state
            if done:
                 self.state, _ = self.env.reset()
                
        return data

    def sync_weights(self, weights):
        self.agent.load_state_dict(weights)

class DummySpace:
    def __init__(self, shape, low, high):
        self.shape = shape
        self.low = low
        self.high = high

def train_parallel():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Master Trainer Device: {device}")
    
    # --- Master Agent Setup ---
    # We create a dummy action space to initialize the Master Agent
    # assuming continuous control as per config
    fwd_max = args.max_forward_speed
    z_max = args.max_vertical_speed
    yaw_max = args.max_yaw_rate
    
    dummy_action_space = DummySpace(
        shape=(3,),
        low=np.array([args.min_forward_speed, -z_max, -yaw_max]),
        high=np.array([fwd_max, z_max, yaw_max])
    )
    
    # Dummy observation shapes
    # (Stack_frames, H, W) -> e.g. (4, 112, 112)
    # config uses 112 or args.image_height? 
    # Let's trust args.image_height if exists, else 112
    h = getattr(args, 'image_height', 112)
    w = getattr(args, 'image_width', 112)
    stack = 4 # Default in AirSimEnv
    
    dummy_depth_shape = (stack, h, w)
    base_dim = 8

    algo = getattr(args, 'algorithm_name', 'gru_td3')
    print(f"Initializing Master Agent: {algo}")
    
    if algo == 'td3':
        learner_agent = TD3Agent(base_dim, dummy_depth_shape, dummy_action_space, args, device)
    else:
        learner_agent = GRUTD3Agent(base_dim, dummy_depth_shape, dummy_action_space, args, device)

    # --- Workers Setup ---
    num_workers = args.n_training_threads
    base_port = args.airsim_port
    
    workers = []
    print(f"Spawning {num_workers} workers starting from port {base_port}...")
    for i in range(num_workers):
        port = base_port + i
        seed = args.seed + i
        w = RolloutWorker.remote(i, port, seed, args, device='cpu')
        workers.append(w)
        
    # --- Training Loop ---
    total_timesteps = args.max_timesteps
    steps_per_worker = 50
    global_step = 0
    
    print("Starting training loop...")
    while global_step < total_timesteps:
        # 1. Collect Data
        futures = [w.sample.remote(steps_per_worker) for w in workers]
        results = ray.get(futures)
        
        # 2. Add to Replay Buffer
        for worker_data in results:
            for t in worker_data:
                state, action, reward, next_state, done = t
                # agent.replay_buffer.add(...)
                # TODO: Ensure your agent.replay_buffer.add signature matches this
                # learner_agent.replay_buffer.add(state, action, reward, next_state, done)
                pass 
            
            global_step += len(worker_data)
        
        # 3. Train
        if global_step > args.start_timesteps:
            learner_agent.train() # or learner_agent.update(args.batch_size) 
            
            # 4. Sync Weights
            if global_step % 1000 < (num_workers * steps_per_worker):
                weights = learner_agent.state_dict()
                ray.get([w.sync_weights.remote(weights) for w in workers])
        
        print(f"Step: {global_step}/{total_timesteps}", end='\r')
        
    print("\nTraining Finished.")
    ray.shutdown()

if __name__ == "__main__":
    train_parallel()
