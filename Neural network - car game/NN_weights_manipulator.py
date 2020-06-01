from algos import Actor, Critic, QFunc
import numpy as np
import tensorflow as tf

from homegym import CarGameEnv

load_name = "DQN_17-52-49_2.00_364"
environment = CarGameEnv(False)
replace_style = 2

# create blank model
# model_new = Actor(environment.observation_space.shape, environment.action_space.high.size, [16, 8])
# model_new = Critic(environment.observation_space.shape, environment.action_space.high.size, [16, 8])
model_new = QFunc(environment.observation_space.shape, environment.action_space.n, [16, 8], True)
print(model_new)
# load existing model
model_load = tf.keras.models.load_model(load_name)
print(model_load)
# calculate new weights
model_weights = [i.numpy() for i in model_load.weights]

for ind, dup in zip([0], [5]):
    temp_weights_dist, temp_weights_misc = np.array_split(model_weights[ind],[len(environment.sense_dist)], axis=0)
    print(model_weights[ind].shape, temp_weights_dist.shape, temp_weights_misc.shape)
    temp_weights_dist = np.tile(temp_weights_dist,(dup,1))/dup
    print(temp_weights_dist.shape)
    model_weights[ind] = np.concatenate((temp_weights_dist, temp_weights_misc), axis=0)

# apply new weights
print([i.shape for i in model_weights])
model = model_new
print(model)
for i in range(len(model.weights)):
    model.weights[i].assign(model_weights[i])
model.save("NEW_" + load_name)