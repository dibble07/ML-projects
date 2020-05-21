# Import libraries
from collections import deque
from datetime import datetime
from homegym import BlobEnv, CarGameEnv
import imageio
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
    render_out = []
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
        current_state = new_state

        # render if desired
        if show_in:
            render_out.append(env_in.render())

    return agent_in, env_in.lap_float, env_in.frame_curr, render_out

def evaluate_fun(env_in, agent_in, show_in):
    global incumbent_lap_float, incumbent_frame_count
    # complete episodes
    lap_float_tot = 0
    frame_count_tot = 0
    render_eval_all = []
    for __ in range(episode_eval_dur):
        __, episode_lap_float, episode_frame_count, episode_render = episode_fun(env_in, 0, agent_in, False, show_in)
        lap_float_tot += episode_lap_float
        frame_count_tot += episode_frame_count
        render_eval_all.extend(episode_render)
    lap_float_avg = lap_float_tot/episode_eval_dur
    frame_count_avg = frame_count_tot/episode_eval_dur
    # save model/video
    save_best = False
    if train:
        if incumbent_lap_float == None or incumbent_frame_count == None:
            save_best = True
        elif lap_float_avg > incumbent_lap_float or (lap_float_avg == incumbent_lap_float and frame_count_avg < incumbent_frame_count):
            save_best = True
    prefix = "Best" if train else "Test"
    filename = f"{prefix}_{datetime.now():%H-%M-%S}_{lap_float_avg:.2f}_{episode_frame_count:3.1f}"
    if save_best:
        print("Saving " + filename)
        incumbent_lap_float, incumbent_frame_count = lap_float_avg, frame_count_avg
        agent.model.save(filename + ".model")
    if (save_best or not train) and len(render_eval_all) > 0:
            with imageio.get_writer(filename + ".mp4", fps=30) as video:
                for render_eval in render_eval_all:
                    video.append_data(render_eval)
    return lap_float_avg, frame_count_avg

class DQN_agent:

    def __init__(self, name):
        self.model = self.create_model(name)
        self.target_model = self.create_model(name)
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=replay_memory_sz)
        self.target_update_counter = 0

    def create_model(self, name):
        if name is None:
            model = Sequential()
            model.add(Dense(64, input_shape=(environment.state.size,)))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(len(environment.action_space), activation='linear'))
            model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        else:
            from keras.models import load_model
            model = load_model(name)

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
        return self.model.predict(state.reshape(-1,state.size))[0]

# Initialise variables and environment
train = True

episode_final = 3_000
episode_eval_freq = 25
episode_eval_dur = 1
eval_vis = True

epsilon_init_ep = 3_000
epsilon_init = 0.1
epsilon_final = 0.01
epsilon_decay = 0.001

discount = 0.95

replay_memory_sz = 50_000
replay_memory_sz_min = 2_000
minibatch_sz = 64
update_target_freq = 5

environment = CarGameEnv()
# environment = BlobEnv(5)
agent = DQN_agent("Best_18-43-23_2.002_358.model")

# Loop for each episode
evaluation_episodes = []
evaluation_lap_float = []
evaluation_frame_count = []
incumbent_lap_float = None
incumbent_frame_count = None

if train:
    for episode in range(episode_final):

        # run episode
        epsilon = epsilon_fun(episode)
        agent, __, __, __ = episode_fun(environment, epsilon, agent, True, False)

        # evaluate
        if not episode % episode_eval_freq or episode+1 == episode_final:
            # run evaluation
            lap_float_avg, frame_count_avg = evaluate_fun(environment, agent, eval_vis)
            # print and save stats of current evaluation
            print(f"{datetime.now():%H:%M:%S} Ep {episode}: train={len(agent.replay_memory) >= replay_memory_sz_min}, epsilon={epsilon_fun(episode):.3f}, reward={lap_float_avg:.2f}, frames={frame_count_avg:3.1f}")
            evaluation_lap_float.append(lap_float_avg)
            evaluation_frame_count.append(frame_count_avg)
            evaluation_episodes.append(episode)

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
    ax2.plot(evaluation_episodes, evaluation_lap_float, color=color)
    ax2.plot(evaluation_episodes, evaluation_frame_count, color=color)
    ax2.set_ylabel(f"Reward", color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.show()
else:
    __ = evaluate_fun(environment, agent, True)