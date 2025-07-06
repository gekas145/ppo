import ale_py
import numpy as np
import gymnasium as gym
from utils import FrameSkip
from gymnasium.wrappers import FrameStackObservation, TransformObservation, TransformAction, ClipReward, ClipAction

def get_racing_env(frame_skip=4, max_ep_len=2000, size=4, randomize=False, clip=True, render_mode=None):
    
    def preprocess(observations):
        observations = observations[:, :84, 6:90, :].astype(np.float32)
        observations = np.mean(observations, axis=-1)
        observations /= 255.
        return observations
    
    def transform_action(act):
        return np.append(act[0], -act[0, 1])
    
    env = gym.make("CarRacing-v3",
                   max_episode_steps=max_ep_len,
                   domain_randomize=randomize,
                   render_mode=render_mode,
                   continuous=True)
    env = ClipAction(env)
    env = TransformAction(env, transform_action, gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32))
    env = FrameSkip(env, frame_skip=frame_skip)
    env = FrameStackObservation(env, stack_size=size)
    env = TransformObservation(env, 
                               func=preprocess,
                               observation_space=gym.spaces.Box(0., 1., shape=(size, 84, 84), dtype=np.float32))
    if clip:
        env = ClipReward(env, -1., 1.)
    return env


def get_pong_env(frame_skip=4, repeat_prob=0.25, max_ep_len=1000, render_mode=None):
    
    def preprocess(observations):
        observations = observations[:, 35:195, :]
        observations = observations[:, ::2,::2].astype(np.float32)
        observations[observations==87] = 0.
        observations[observations!=0] = 1.
        observations = observations[0] - observations[1]
        return observations[np.newaxis, ...]

    def transform_action(act):
        act = act[0]
        if act == 0:
           return 2
        return 3
    
    env = gym.make("ALE/Pong-v5", 
                   frameskip=frame_skip, 
                   obs_type='grayscale', 
                   repeat_action_probability=repeat_prob,
                   render_mode=render_mode,
                   max_episode_steps=max_ep_len)
    env = TransformAction(env, transform_action, gym.spaces.Box(0., 1., shape=(1,1), dtype=np.float32))
    env = FrameStackObservation(env, stack_size=2)
    env = TransformObservation(env, 
                               func=preprocess,
                               observation_space=gym.spaces.Box(-1.0, 1.0, shape=(1, 80, 80), dtype=np.float32))
    return env


if __name__ == '__main__':
    env = get_racing_env()
    print(env.observation_space._shape)
    print(env.action_space._shape)
    env.reset()
    env.step(np.array([[-2.5, 3.]]))
