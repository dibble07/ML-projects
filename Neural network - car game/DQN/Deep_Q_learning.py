# Import libraries
from collections import deque
from homegym import BlobEnv, CarGameEnv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import numpy as np
np.random.seed(42)
import os
import random
random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

print("""
    To do:
Redo model saving policy
""")

# Define user functions/classes
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

def episode_fun(env_in, epsilon_in, agent_in, update_Q, show_in):
    # reset episode
    current_state = env_in.reset()
    episode_reward = 0
    step_count = 0
    done = False
    while not done:

        # take action
        if np.random.random() > epsilon_in:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, len(env_in.action_space))
        new_state, reward, done = env_in.step(action)

        # update Q model
        if update_Q:
            agent_in.update_replay_memory((current_state, action, reward, new_state, done))
            agent_in.train(done)

        # prepare for next step
        episode_reward += reward
        current_state = new_state

        # render if desired
        if show_in:
            env_in.render()

    return agent_in, episode_reward

class DQN_agent:

    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=replay_memory_sz)
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(len(environment.observation_space),)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(len(environment.action_space), activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state):

        if len(self.replay_memory) >= replay_memory_sz_min:

            # Get a minibatch and associated Q values
            minibatch = random.sample(self.replay_memory, minibatch_sz)
            current_states = np.array([transition[0] for transition in minibatch])
            current_qs_list = self.model.predict(current_states)
            new_current_states = np.array([transition[3] for transition in minibatch])
            future_qs_list = self.target_model.predict(new_current_states)

            # Loop through experiences
            X = []
            y = []
            for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
                if not done:
                    max_future_q = np.max(future_qs_list[index])
                    new_q = reward + discount * max_future_q
                else:
                    new_q = reward
                current_qs = current_qs_list[index]
                current_qs[action] = new_q
                X.append(current_state)
                y.append(current_qs)

            # Fit model
            if terminal_state:
                self.model.fit(np.array(X), np.array(y), batch_size=minibatch_sz, verbose=0, shuffle=False)
                self.target_update_counter += 1

            # Update target network
            if self.target_update_counter >= update_target_freq:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

        return

    def get_qs(self, state):
        return self.model.predict(np.asarray(state).reshape(-1, len(environment.observation_space)))[0]

# Initialise variables and environment
episode_final = 20_000
episode_eval_freq = 100
episode_eval_dur = 1
episode_length = float('inf')
eval_vis = True

epsilon_init_ep = 5_000
epsilon_init = 0.1
epsilon_final = 0.01
epsilon_decay = 0.001

discount = 0.95

replay_memory_sz = 10_000
replay_memory_sz_min = 1_000
minibatch_sz = 64
update_target_freq = 5

environment = CarGameEnv()
# environment = BlobEnv(5)
agent = DQN_agent()

# Loop for each episode
evaluation_episodes = []
evaluation_rewards = []
for episode in range(episode_final):

    # run episode
    epsilon = epsilon_fun(episode)
    agent, __ = episode_fun(environment, epsilon, agent, True, False)

    # evaluate
    if not episode % episode_eval_freq or episode+1 == episode_final:
        reward_eval_tot = 0
        for __ in range(episode_eval_dur):
            __, episode_reward = episode_fun(environment, 0, agent, False, eval_vis)
            reward_eval_tot += episode_reward
        reward_eval_avg = reward_eval_tot/episode_eval_dur
        print(f"Episode {episode}: epsilon is {epsilon_fun(episode):.3f}, reward is {reward_eval_avg:.3f}")
        evaluation_rewards.append(reward_eval_avg)
        evaluation_episodes.append(episode)
        # Save model, but only when min reward is greater or equal a set value
        # if min_reward[-1] >= MIN_REWARD:
        #     agent.model.save(f'models/{MODEL_NAME}__{average_reward[-1]:_>7.2f}avg_{std_dev_reward[-1]:_>7.2f}std_dev__{int(time.time())}.model')

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