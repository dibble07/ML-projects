# Import libraries
print("Importing libraries")
from geneticoptimiser import floatopt
import connectfour
from deap import base
from deap import creator
from deap import tools
import keras
import numpy as np
import random
random.seed(42)

# Define user functions

def NeuralNetwork():
	model_out = keras.models.Sequential()
	model_out.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(6,7,2)))
	model_out.add(keras.layers.Conv2D(8, kernel_size=3, activation='relu'))
	model_out.add(keras.layers.Flatten())
	model_out.add(keras.layers.Dense(7, activation='softmax'))
	return model_out

def ManyGames(user_type_in, model_in, play_start):
	outcome_tot={0:0, 1:0, 2:0}
	for i in range(no_games):
		__, game_result = connectfour.PlayGame(user_type_in, model_in, play_start)
		outcome_tot[game_result]+=1
	return outcome_tot

def ObjFun(weights_vec):
	# set model weights - split weightings, normalise and offset, assign to Neural Network
	weights_vec = np.array(weights_vec)
	weights_norm = np.add(np.multiply(weights_vec,weight_mult), weight_add)
	weights_vec_split=np.split(weights_norm,np.cumsum(weights_size))
	weights_arr = [mat.reshape(shape) for mat, shape in zip(weights_vec_split[:-1], weights_shape)]
	model_untrained.set_weights(weights_arr)
	outcomes_all = {1: 0, 2: 0}
	# player 1 goes first
	outcomes = ManyGames(types_in, model_untrained, 1)
	outcomes_all[1]+=outcomes[1]
	outcomes_all[2]+=outcomes[2]
	# player 2 goes first
	outcomes = ManyGames(types_in, model_untrained, 2)
	outcomes_all[2]+=outcomes[1]
	outcomes_all[1]+=outcomes[2]
	# calulate score
	score = (outcomes_all[1] - outcomes_all[2])/(2*no_games)
	return score,


print("""
To do:
	check initilisation logic for convolutional network with relu
	chech double import of random
	""")

# Define neural network
model_untrained = NeuralNetwork()
weights_shape = [mat.shape for mat in model_untrained.get_weights()]
weights_size = [mat.size for mat in model_untrained.get_weights()]
norm_list=[]
for i in range(int(len(weights_size)/2)):
	norm_list.append(np.ones((sum(weights_size[2*i:2*(i+1)])))*2/(weights_size[2*i])**0.5)
weight_mult = np.concatenate(norm_list)
weight_add = np.zeros(weight_mult.shape)

# Define optimisaion parameters
mut_prob = 0.2
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, sum(weights_size))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", ObjFun)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
toolbox.register("select", tools.selBest)

# First round of optimisation
types_in = {1: 'NeuralNetwork', 2:'Rand'}
pop_size = 16
no_games = 50  
max_gen = 1000
max_obj = 0.4
ev_best = floatopt(toolbox, tools, pop_size, mut_prob, max_gen, max_obj)
best = ev_best[np.argmax([ev.fitness.values[0] for ev in ev_best])]
best=[np.array(best),best.fitness.values[0]]
weight_add = best[0]
print(f"First round achieved a score of {best[1]}")

# Second round of optimisation
types_in = {1: 'NeuralNetwork', 2:'Rand'}
pop_size = 32
no_games = 50  
max_gen = 1000
max_obj = 0.65
ev_best = floatopt(toolbox, tools, pop_size, mut_prob, max_gen, max_obj)
best = ev_best[np.argmax([ev.fitness.values[0] for ev in ev_best])]
best=[np.array(best),best.fitness.values[0]]
weight_add = best[0]
print(f"Second round achieved a score of {best[1]}")
