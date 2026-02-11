import numpy as np
import torch


class OUNoise:
    """Ornstein-Uhlenbeck process noise generator.

    Usage:
      ou = OUNoise(size=action_dim, mu=0.0, theta=0.15, sigma=0.2, sigma_min=0.01)
      action_noise = ou.sample()            # numpy array shaped (action_dim,)
      t = ou.sample_tensor(shape=(batch, action_dim), device=device)
    """

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1.0, x0=None, sigma_min=None, sigma_decay=None):
        if isinstance(size, int):
            self.size = (size,)
        else:
            self.size = tuple(size)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.initial_sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        if self.x0 is not None:
            arr = np.array(self.x0, dtype=np.float32)
            self.x_prev = arr.reshape(self.size)
        else:
            self.x_prev = np.ones(self.size, dtype=np.float32) * np.float32(self.mu)

    def scale_sigma(self, decay_ratio):
        """
        Explicitly scale sigma based on a decay ratio (e.g. from training progress).
        Uses sigmoid-like decay: 前期慢，中期快，后期平缓
        
        Args:
            decay_ratio: float between 0 and 1, representing training progress
        """
        # Sigmoid-like decay: 前期慢，中期快，后期平缓
        # 使用调整的sigmoid函数，从初始值衰减到最小值
        x = 10 * (decay_ratio - 0.6)  # 调整sigmoid的中心和陡度
        sigmoid_factor = 1 / (1 + np.exp(-x))  # 从0到1
        current_sigma = self.initial_sigma - (self.initial_sigma - self.sigma_min) * sigmoid_factor
        
        if self.sigma_min is not None:
            self.sigma = max(self.sigma_min, current_sigma)
        else:
            self.sigma = current_sigma

    def sample(self):
        noise = np.random.normal(size=self.size).astype(np.float32)
        x = self.x_prev + self.theta * (np.float32(self.mu) - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * noise
        self.x_prev = x
        return x

    def sample_tensor(self, shape=None, device=None, dtype=torch.float32):
        """Return a torch tensor noise. If requested shape differs from the OU internal
        shape, produce stateless OU-like samples for that shape (independent draws).
        """
        if shape is None or tuple(shape) == self.size:
            arr = self.sample()
        else:
            noise = np.random.normal(size=tuple(shape)).astype(np.float32)
            mu_arr = np.float32(self.mu)
            arr = mu_arr + self.sigma * np.sqrt(self.dt) * noise

        t = torch.from_numpy(arr)
        if device is not None:
            t = t.to(device)
        return t.type(dtype)
