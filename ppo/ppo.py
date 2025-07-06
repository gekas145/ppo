import time
import torch
import utils as ut
import numpy as np
from tqdm import tqdm
from gymnasium import Env
from collections import deque
from torch.optim.lr_scheduler import LambdaLR


def ppo(policy_type: str, 
        model: torch.Module, 
        env: Env, 
        test_env: Env = None,
        sample_steps: int = 1024,
        epochs: int = 50,
        epoch_rounds: int = 25,
        batch_size: int = 64,
        opt_steps: int = 10,
        lr: float = 1e-4,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        eps: float = 0.1,
        log_scale_max: float = -1.1,
        log_scale_min: float = -2.3,
        critic_coef: float = 0.5,
        test_runs: int = 10,
        early_stopping: int = -1
        ) -> None:
    
    """
    Runs Policy Proximal Optimization algorithm.

    Args:
    policy_type    -- the type of policy that model will parametrize; can be 'discrete'(Categorical) or 'continuous'(Normal)
    model          -- instance of torch.Module class; is expected to return tuple of policy parameters and critic value given enviroment state
    env            -- enviroment used to sample training data
    test_env       -- enviroment used at test runs; if None is same as env
    sample_steps   -- # of steps to be collected for learning
    epochs         -- # of epoch_rounds series during algorithm run
    epoch_rounds   -- # of learning rounds in one epoch 
    batch_size     -- size of batch in optimization algorithm
    opt_steps      -- # of full passes through collected data during one learning round
    lr             -- learning rate in optimization algorithm
    gamma          -- discount factor for General Advantage Estimation
    lmbda          -- discount factor for rewards
    eps            -- epsilon parameter of clipped loss
    log_scale_max  -- log of scale of normal distribution is linearly annealed up from this value; used only when policy_type='continuous'
    log_scale_min  -- log of scale of normal distribution is linearly annealed down to this value; used only when policy_type='continuous'
    critic_coef    -- weight of critic loss in overall loss
    test_runs      -- # of full runs through test_env on epoch end
    early_stopping -- # of epochs without update of best running reward to stop algorithm after; set to -1 to disable
    """
    
    @torch.no_grad
    def step(obs, scale):
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        actor_output, critic_output = model(obs_tensor)
        act, act_logp = ut.sample_actions(actor_output, scale, policy_type)
        return act.numpy(), act_logp.numpy(), critic_output.numpy()

    def get_batch(batch_id, train_data):
        ids = batch_indexes[batch_id*batch_size:(batch_id + 1)*batch_size]
        return {k: torch.from_numpy(train_data[k][ids,]) for k in train_data.keys()}
    
    if policy_type != 'discrete' and policy_type != 'continuous':
        raise ValueError(f'Unknown value of policy_type: {policy_type}')
    
    if test_env is None:
        test_env = env

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lambda step: 1. - step/(epochs * epoch_rounds * opt_steps))
    buffer = ut.PPOBuffer(env.observation_space._shape, env.action_space._shape, sample_steps, 
                          gamma=gamma, lam=lmbda)
    batch_indexes = np.array(range(sample_steps))
    rewards_stack = deque([], maxlen=100)
    best_running_reward = -np.inf
    log_scale = log_scale_max
    scale = np.exp(log_scale)


    for epoch in range(1, epochs+1):
        start_time = time.time()

        for epoch_round in tqdm(range(epoch_rounds), leave=False, ncols=50):
        
            obs, _ = env.reset()
            # collect trajectories
            for i in range(sample_steps):
                act, act_logp, val = step(obs, scale)
                obs_new, rew, terminated, truncated, _ = env.step(act)
                buffer.store(obs, act, rew, val, act_logp)
                obs = obs_new

                if terminated or truncated or i == sample_steps-1:
                    if terminated:
                        val = 0.
                    else:
                        _, _, val = step(obs, scale)
                    
                    buffer.finish_path(val)
                    obs, _ = env.reset()


            train_data = buffer.get()

            # run optimization
            for _ in range(opt_steps):
                np.random.shuffle(batch_indexes)
                for batch in range(sample_steps//batch_size):
                    optimizer.zero_grad()
                    train_data_batch = get_batch(batch, train_data)
                    actor_output, critic_output = model(train_data_batch['obs'])

                    distr = ut.get_distr(actor_output, scale, policy_type)
                    act_logp = distr.log_prob(np.squeeze(train_data_batch['act']))
                    if len(act_logp.shape) != 1:
                        act_logp = act_logp.sum(axis=-1)
                    
                    act_pratios = torch.exp(act_logp - train_data_batch['logp'])
                    actor_loss = torch.min(act_pratios * train_data_batch['adv'],
                                           torch.clip(act_pratios, 1. - eps, 1. + eps) * train_data_batch['adv'])
                    actor_loss = -actor_loss.mean()

                    critic_loss = torch.mean((critic_output.ravel() - train_data_batch['ret'])**2)

                    loss = actor_loss + critic_loss * critic_coef
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
            
            log_scale -= (log_scale_max - log_scale_min)/(epochs * epoch_rounds)
            scale = np.exp(log_scale)

        # test performance
        for i in range(test_runs):
            total_rew = 0
            test_obs, _ = test_env.reset()
            while True:
                
                act, _, _ = step(test_obs, scale)
                test_obs, rew, terminated, truncated, _ = test_env.step(act)
                total_rew += rew

                if truncated or terminated:
                    rewards_stack.append(total_rew)
                    break
            
        # calculate current running reward and print some info
        running_reward = np.mean(rewards_stack)
        if running_reward > best_running_reward:
            early_stopping_count = 0
            best_running_reward = running_reward
            ut.save_weights('../models/model.pt', model)
        else:
            early_stopping_count += 1

        message = f'[epoch {epoch}] '
        message += f'running reward: {running_reward:.2f}, '
        message += f'best reward: {best_running_reward:.2f}, '
        message += f'frames: {sample_steps*epoch*epoch_rounds}, '
        if policy_type != 'discrete':
            message += f'scale: {scale:.2f}, '
        message += f'time: {(time.time() - start_time):.2f}'
        print(message)

        if early_stopping_count >= early_stopping and early_stopping != -1:
            print(f'No reward improvement for {early_stopping_count} epochs, exiting early')
            break
