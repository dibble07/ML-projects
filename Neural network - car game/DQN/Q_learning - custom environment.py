# import libraries
import cv2
from homegym import BlobEnv
from matplotlib import style
style.use("ggplot")
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
import time

print("""
    To do:
Initialise with None/NaN
Evaluate with epsilon equal 0
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

def episode_fun(env_in, epsilon_in, Q_table_in, show_in):
    # reset episode
    obs = env_in.reset()
    obs_ind = tuple(i+size-1 for i in obs)
    episode_reward = 0
    step_count = 0
    done = False
    while not done:
        # take action
        if np.random.random() > epsilon_in:
            # print(obs)
            # print(tuple(i+size-1 for i in obs))
            # print(Q_table_in[obs])
            action = np.argmax(Q_table_in[obs_ind])
        else:
            action = np.random.randint(0, 4)
        new_obs, reward, done = env_in.step(action)
        new_obs_ind = tuple(i+size-1 for i in new_obs)

        # update Q table
        max_future_q = np.max(Q_table_in[new_obs_ind])
        current_q = Q_table_in[obs_ind][action]
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
        Q_table_in[obs_ind][action] = new_q

        # prepare for next episode
        episode_reward += reward
        if step_count < episode_length:
            step_count += 1
        else:
            done = True
        obs = new_obs
        obs_ind = tuple(i+size-1 for i in obs)


        # render if desired
        if show_in:
            environment.render()

    return Q_table_in, episode_reward

# Initialise variables and environment
episode_final = 12_000
episode_vis = 1_000
episode_length = 100

epsilon_init = 0.5
epsilon_init_ep = 6_000
epsilon_decay = 0.001
epsilon_final = 0.05

learning_rate = 0.1
discount = 0.95
size = 6
environment = BlobEnv(size)

# Initialise Q table
load_Q_table = None
if load_Q_table is None:
    Q_table = np.random.uniform(-5, 0, (2*size-1,)*4 + (4,))
else:
    with open(load_Q_table, "rb") as f:
        Q_table = pickle.load(f)

# Loop for each episode
episode_rewards = []
moving_avg = []
for episode in range(episode_final):
    
    # display this episode
    show = True if not episode % episode_vis and episode > 0 else False

    # run episode
    epsilon = epsilon_fun(episode)
    Q_table, episode_reward = episode_fun(environment, epsilon, Q_table, show)

    # save and display analytics
    episode_rewards.append(episode_reward)
    moving_avg.append(np.mean(episode_rewards[-episode_vis:]))
    if show:
        print(f"Episode {episode}: epsilon is {epsilon:.3f}, {episode_vis} ep mean is {moving_avg[-1]}")

# Visulaise results
eps = list(range(episode_final))
fig, ax1 = plt.subplots()
ax1.set_xlabel("Episode")
color = 'tab:red'
ax1.plot(eps, epsilon_fun(eps), color=color)
ax1.set_ylabel("Epsilon", color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(eps, moving_avg, color=color)
ax2.set_ylabel(f"Reward {episode_vis}ma", color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.show()

# with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#     pickle.dump(Q_table, f)