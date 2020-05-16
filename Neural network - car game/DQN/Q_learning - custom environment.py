# Import libraries
import cv2
from homegym import BlobEnv, CarGameEnv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import numpy as np
import pickle
from PIL import Image
import time

print("""
    To do:
Plot performance at start and update figure 
""")

# Define user functions
def epsilon_fun(eps_in):
    if isinstance(eps_in, list):
        eps = eps_in
    else:
        eps = [eps_in]
    epsilon = []
    eps_range = epsilon_init - epsilon_final
    for ep in eps:
        if ep < epsilon_init_ep:
            epsilon.append(0)
        else:
            epsilon.append(eps_range*np.exp(-epsilon_decay*(ep-epsilon_init_ep))+epsilon_final)
    output = epsilon[0] if len(epsilon) == 1 else epsilon
    return output

def episode_fun(env_in, epsilon_in, Q_table_in, update_Q, show_in):
    # reset episode
    obs = env_in.reset()
    obs_ind = tuple(np.argmin(np.abs(y-x)) for x, y in zip(obs, env_in.observation_space))
    episode_reward = 0
    step_count = 0
    done = False
    while not done:
        # take action
        if np.random.random() > epsilon_in:
            Q_table_obs = Q_table_in[obs_ind]
            incomplete = np.isnan(Q_table_obs)
            if any(incomplete) and update_Q:
                incomplete_ind = np.where(incomplete)[0]
                if incomplete_ind.size>1:
                    action = np.random.choice(incomplete_ind)
                else:
                    action = incomplete_ind[0]
            else:
                action = np.argmax(Q_table_obs)
        else:
            action = np.random.randint(0, len(env_in.action_space))
        new_obs, reward, done = env_in.step(action)
        new_obs_ind = tuple(np.argmin(np.abs(y-x)) for x, y in zip(new_obs, env_in.observation_space))

        # update Q table
        if update_Q:
            max_future_q = np.max(Q_table_in[new_obs_ind])
            current_q = Q_table_in[obs_ind][action]
            if np.isnan(max_future_q):
                delta_q = reward
            else:
                delta_q = reward + discount * max_future_q
            if np.isnan(current_q):
                new_q = delta_q
            else:
                new_q = (1 - learning_rate) * current_q + learning_rate * delta_q
            Q_table_in[obs_ind][action] = new_q

        # prepare for next step
        episode_reward += reward
        if step_count < episode_length:
            step_count += 1
        else:
            done = True
        obs_ind = new_obs_ind

        # render if desired
        if show_in:
            env_in.render()

    return Q_table_in, episode_reward

# Initialise variables and environment
episode_final = 6_000
episode_eval_freq = 100
episode_eval_dur = 1
episode_length = float('inf')
eval_vis = True

epsilon_init_ep = 1_000
epsilon_init = 0.05
epsilon_final = 0.01
epsilon_decay = 0.001

learning_rate = 0.1
discount = 0.95
environment = CarGameEnv()
# environment = BlobEnv(5)

# Initialise Q table
load_Q_table = None#"Q_table.npy"
try:
    Q_table = np.load(load_Q_table)
except:
    print("No Q table loaded")
    Q_table = np.empty(tuple(i.size for i in environment.observation_space) + (len(environment.action_space),))
    Q_table[:] = np.NaN

# Loop for each episode
evaluation_episodes = []
evaluation_rewards = []
for episode in range(episode_final):
    
    # run episode
    epsilon = epsilon_fun(episode)
    Q_table, __ = episode_fun(environment, epsilon, Q_table, True, False)

    # evaluate
    if not episode % episode_eval_freq:
        reward_eval_tot = 0
        for __ in range(episode_eval_dur):
            __, episode_reward = episode_fun(environment, 0, Q_table, False, eval_vis)
            reward_eval_tot += episode_reward
        reward_eval_avg = reward_eval_tot/episode_eval_dur
        print(f"Episode {episode}: epsilon is {epsilon_fun(episode):.3f}, reward is {reward_eval_avg:.3f}")
        evaluation_rewards.append(reward_eval_avg)
        evaluation_episodes.append(episode)
        np.save("Q_table", Q_table)

# Visualise results
print("All episodes done")
eps = list(range(episode_final))
fig, ax1 = plt.subplots()
ax1.set_xlabel("Episode")
color = 'tab:red'
ax1.plot(eps, epsilon_fun(eps), color=color)
ax1.set_ylabel("Epsilon", color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(evaluation_episodes, evaluation_rewards, color=color)
ax2.set_ylabel(f"Reward", color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.show()