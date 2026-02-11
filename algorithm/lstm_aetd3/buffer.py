import numpy as np

class SequenceReplayBuffer:
    """Replay buffer that stores and samples sequences of transitions for POMDP.
    Optimized with pre-allocated numpy arrays for faster sampling.
    """

    def __init__(self, max_size: int, sequence_length: int):
        self.max_size = int(max_size)
        self.seq_len = int(sequence_length)
        self.ptr = 0
        self.size = 0
        
        # Pre-allocated buffers (initialized on first add)
        self.base_buf = None
        self.depth_buf = None
        self.gray_buf = None
        self.action_buf = None
        self.reward_buf = None
        self.done_buf = None

    def add(self, base_state, depth, action, reward, done):
        """
        Add a single step.
        """
        # Lazy initialization of buffers
        if self.base_buf is None:
            self.base_dim = base_state.shape[0]
            self.depth_shape = depth.shape # (1, H, W)
            self.action_dim = action.shape[0]
            
            self.base_buf = np.zeros((self.max_size, self.base_dim), dtype=np.float32)
            self.depth_buf = np.zeros((self.max_size, *self.depth_shape), dtype=np.float16) # Optimize: Store depth as float16
            self.action_buf = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
            self.reward_buf = np.zeros((self.max_size, 1), dtype=np.float32)
            self.done_buf = np.zeros((self.max_size, 1), dtype=np.float32)

        # Store data
        self.base_buf[self.ptr] = base_state
        self.depth_buf[self.ptr] = depth.astype(np.float16)
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        """
        Sample batch_size sequences of length seq_len.
        Returns tensors of shape (B, K, ...).
        """
        # Sample logical indices: [0, size - seq_len]
        # Logical index 0 corresponds to the oldest element in the buffer
        # Logical index size-1 corresponds to the newest element
        
        # We need sequences of length seq_len + 1 (for next_state)
        # So we can only sample up to size - seq_len - 1?
        # Actually, we need indices i, i+1, ..., i+K-1 for state
        # And i+1, ..., i+K for next_state
        # So we need valid data up to i+K.
        # So logical start index can go up to size - seq_len - 1.
        # Wait, if we have size elements, valid indices are 0..size-1.
        # We need index+seq_len to be <= size-1.
        # So index <= size - 1 - seq_len.
        
        if self.size <= self.seq_len:
            # Not enough data
            return None 
            
        valid_max_idx = self.size - self.seq_len
        
        batch_indices = []
        
        # Optimistic sampling with rejection
        while len(batch_indices) < batch_size:
            remaining = batch_size - len(batch_indices)
            # Sample more than needed to account for rejections
            candidates = np.random.randint(0, valid_max_idx, size=remaining * 2)
            
            # Convert logical to physical indices
            # Physical = (ptr - size + logical) % max_size
            # Note: ptr points to the *next* empty slot, so ptr-size is the oldest.
            start_idxs = (self.ptr - self.size + candidates) % self.max_size
            
            # Check validity: no 'done' in the first K-1 steps of the sequence
            # We want the sequence to be in the same episode.
            # Sequence indices: start, start+1, ..., start+seq_len-1
            # If any of start...start+seq_len-2 is done, then the sequence crosses episode.
            # We check done_buf.
            
            is_valid = np.ones(len(candidates), dtype=bool)
            
            # Vectorized check for done flags
            # We need to check offsets 0, 1, ..., seq_len-2
            for k in range(self.seq_len - 1):
                check_idxs = (start_idxs + k) % self.max_size
                # If done_buf is 1, it's a terminal state.
                # If step k is terminal, then step k+1 is a new episode.
                # So the sequence is invalid.
                dones = self.done_buf[check_idxs].squeeze()
                is_valid = is_valid & (dones == 0)
                
            valid_candidates = candidates[is_valid]
            
            # Add valid physical start indices to batch
            valid_start_idxs = (self.ptr - self.size + valid_candidates) % self.max_size
            batch_indices.extend(valid_start_idxs[:remaining])

        batch_indices = np.array(batch_indices[:batch_size])
        
        # Construct full sequence indices (B, K)
        # We need state sequence (0..K-1) and next_state sequence (1..K)
        # Let's construct indices for 0..K
        
        # shape: (B, K+1)
        seq_indices = (batch_indices[:, None] + np.arange(self.seq_len + 1)) % self.max_size
        
        # Retrieve data
        # (B, K+1, ...)
        bases = self.base_buf[seq_indices]
        depths = self.depth_buf[seq_indices].astype(np.float32) # Convert back to float32
        actions = self.action_buf[seq_indices]
        rewards = self.reward_buf[seq_indices]
        dones = self.done_buf[seq_indices]
        
        # Split into state and next_state
        # State: 0..K-1
        batch_base = bases[:, :-1]
        batch_depth = depths[:, :-1]
        batch_action = actions[:, :-1]
        batch_reward = rewards[:, :-1]
        batch_done = dones[:, :-1]
        
        # Next State: 1..K
        batch_next_base = bases[:, 1:]
        batch_next_depth = depths[:, 1:]
        
        return (
            batch_base,
            batch_depth,
            batch_action,
            batch_reward,
            batch_next_base,
            batch_next_depth,
            batch_done
        )

    def size_buffer(self):
        return self.size

    def size(self):
        return self.size

