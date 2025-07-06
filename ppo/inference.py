import torch
import pickle
import utils as ut
import numpy as np
from tqdm import tqdm
from model import BigModel
from enviroment import get_racing_env

num_episodes = 100
model = BigModel(4, 2)
with open('../models/model_car.pt', 'rb') as f:
    state_dict = pickle.load(f)
model.load_state_dict(state_dict)
model.eval()

env = get_racing_env(max_ep_len=1500, clip=False)

rewards = [0] * num_episodes
for i in tqdm(range(num_episodes), ncols=50):
    episode_over = False
    episode_reward = 0
    observation, _ = env.reset()
    while not episode_over:
        with torch.inference_mode():
            input = torch.from_numpy(observation)
            input = input.unsqueeze(0)
            model_output, _ = model(input)

        action, _ = ut.sample_actions(model_output, 0.1, 'continuous')

        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_over = terminated or truncated

    rewards[i] = episode_reward

print('Mean reward:', np.mean(rewards))
