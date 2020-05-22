from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from homegym import BlobEnv, CarGameEnv
import numpy as np

load_name = "Best_07-17-40_1.956.model"
environment = CarGameEnv()
replace_style = 2

# create blank model
model_new = Sequential()
model_new.add(Dense(64, input_shape=(environment.state.size,)))
model_new.add(Dense(32, activation='relu'))
model_new.add(Dense(len(environment.action_space), activation='linear'))
model_new.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
# load existing model
model_load = load_model(load_name)
# calculate new weights
model_new_weights = model_new.get_weights()
print([i.shape for i in model_new_weights])
model_load_weights = model_load.get_weights()
print([i.shape for i in model_load_weights])
if replace_style == 1:
    new_weights = []
    for new_layer, new_bias, load_layer, load_bias in zip(model_new_weights[::2], model_new_weights[1::2], model_load_weights[::2], model_load_weights[1::2]):
        updated_layer = new_layer
        updated_bias = new_bias
        i_row, i_col = load_layer.shape
        updated_layer[0:i_row,0:i_col] = load_layer
        updated_bias[0:i_col] = load_bias
        new_weights.append(updated_layer)
        new_weights.append(updated_bias)
elif replace_style == 2:
    new_weights = model_load_weights
    for ind, dup in zip([0], [5]):
        temp_weights_dist, temp_weights_misc = np.array_split(new_weights[ind],[len(environment.sense_dist)], axis=0)
        print(new_weights[ind].shape, temp_weights_dist.shape, temp_weights_misc.shape)
        temp_weights_dist = np.tile(temp_weights_dist,(dup,1))/dup
        print(temp_weights_dist.shape)
        new_weights[ind] = np.concatenate((temp_weights_dist, temp_weights_misc), axis=0)

# apply new weights
print([i.shape for i in new_weights])
model = model_new
model.set_weights(new_weights)
model.save("NEW_" + load_name)