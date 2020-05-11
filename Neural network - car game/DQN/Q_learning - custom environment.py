import cv2
from homegym import BlobEnv
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import numpy as np
import pickle
from PIL import Image
import time

# Define variables
HM_EPISODES = 24_000
epsilon = 0.9
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 3000  # how often to play through env visually.
start_q_table = None # None or Filename
LEARNING_RATE = 0.1
DISCOUNT = 0.95

# Initialise Q table
if start_q_table is None:
    q_table = {}
    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1, SIZE):
                    for iiii in range(-SIZE+1, SIZE):
                            q_table[(i, ii, iii, iiii)] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

# Train and update Q table
episode_rewards = []
environment = BlobEnv()
for episode in range(HM_EPISODES):
    if episode % SHOW_EVERY == 0 and episode > 0:
        print(f"#{episode}: epsilon = {epsilon} , {SHOW_EVERY} ep mean = {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    obs = environment.reset()
    for __ in range(200):
        # get action
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        new_obs, reward, done = environment.step(action)
        episode_reward += reward
        # update Q table
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q
        # visualise
        if show:
            environment.render()
        # break loop if done
        if done:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

# Visualise outputs
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

# with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#     pickle.dump(q_table, f)