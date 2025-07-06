from ppo import ppo
from model import BigModel, SmallModel
from enviroment import get_racing_env, get_pong_env

## train car model
in_channels = 4
model = BigModel(in_channels, 2)
env = get_racing_env(size=in_channels)
test_env = get_racing_env(max_ep_len=800, size=in_channels, clip=False)

ppo('continuous', model, env, test_env=test_env)


## train pong model
# in_channels = 1
# model = SmallModel(in_channels, 1)
# env = get_pong_env()
# test_env = get_pong_env(repeat_prob=0., max_ep_len=8000)


# ppo('discrete', model, env, test_env,
#     sample_steps=1000,
#     epochs=40,
#     epoch_rounds=50,
#     batch_size=1000,
#     opt_steps=12,
#     lr=1e-3,
#     gamma=0.99,
#     lmbda=0.95,
#     eps=0.1,
#     critic_coef=1.,
#     test_runs=10)
