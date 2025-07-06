import torch
import pickle
import numpy as np
from gymnasium import Wrapper
from torch.distributions import Normal, Categorical


class FrameSkip(Wrapper):
     
    def __init__(self, env, frame_skip):
        super().__init__(env)
        if frame_skip < 1:
            raise ValueError('frame_skip has to be positive!')
        self.frame_skip = frame_skip
    
    def step(self, action):
        total_rew = 0.
        for i in range(self.frame_skip):
            obs, rew, terminated, truncated, info = self.env.step(action)
            total_rew += rew
            if terminated or truncated:
                break
        
        return obs, total_rew, terminated, truncated, info

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.

    Borrowed from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((combined_shape(size, obs_dim)), dtype=np.float32)
        self.act_buf = np.zeros((combined_shape(size, act_dim)), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        self.adv_buf = normalize(self.adv_buf)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return data

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def normalize(values):
    mean = np.mean(values)
    std = np.std(values)
    return (values - mean) / std

def discount(values, factor):
    discounted_values = np.zeros_like(values)
    discounted_values[-1] = values[-1]
    for i in range(discounted_values.shape[0]-2, -1, -1):
            discounted_values[i] = factor * discounted_values[i+1] + values[i]

    return discounted_values

def get_distr(actor_output, scale, policy_type):
    if policy_type == 'discrete':
        return Categorical(logits=actor_output)
    return Normal(actor_output, scale)

def sample_actions(actor_output, scale, policy_type):
    distr = get_distr(actor_output, scale, policy_type)
    actions = distr.sample()
    return actions, distr.log_prob(actions).sum(axis=-1)

def save_weights(fname, model):
    with open(fname, "wb") as f:
            pickle.dump(model.state_dict(), f)

def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


if __name__ == '__main__':
    actor_output = torch.normal(mean=0., std=1., size=(1, 2))

    distr = get_distr(actor_output, 0.1, 'continuous')
    act, log_p = sample_actions(actor_output, 0.1, 'continuous')
    print(act.numpy())
    print(log_p - distr.log_prob(act).sum(axis=-1))


