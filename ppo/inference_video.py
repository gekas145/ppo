import torch
import pickle
import utils as ut
from model import BigModel
from enviroment import get_racing_env
from gymnasium.wrappers import RecordVideo


model = BigModel(4, 2)
with open('../models/model_car.pt', 'rb') as f:
    state_dict = pickle.load(f)
model.load_state_dict(state_dict)
model.eval()

env = get_racing_env(render_mode='rgb_array', max_ep_len=1500, randomize=False, clip=False)
env = RecordVideo(env, video_folder='../recordings', fps=16)
observation, _ = env.reset()

total_reward = 0
while True:
    with torch.inference_mode():
        input = torch.from_numpy(observation)
        input = input.unsqueeze(0)
        actor_output, _ = model(input)

    action, _ = ut.sample_actions(actor_output, 0.1, 'continuous')
    action = action.numpy()
    observation, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

env.close()
print('Total reward:', total_reward)
