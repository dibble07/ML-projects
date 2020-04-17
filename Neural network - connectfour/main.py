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

def NeuralNetwork(arch_layers):
	model_out = keras.models.Sequential()

	# if arch_layers[0][0] is "dense":
	# 	model_out.add(keras.layers.Dense(units=arch_layers[0][1], activation=arch_layers[0][2], input_shape=(42,)))
	# for arch_layer in arch_layers[1:]:
	# 	if arch_layer[0] is "dense":
	# 		model_out.add(keras.layers.Dense(units=arch_layer[1], activation=arch_layer[2]))

	model_out.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(6,7,2)))
	model_out.add(keras.layers.Conv2D(8, kernel_size=3, activation='relu'))
	model_out.add(keras.layers.Flatten())
	model_out.add(keras.layers.Dense(7, activation='softmax'))

	return model_out

def ManyGames(user_type_in, model_in, play_start):
	outcome_tot={0:0, 1:0, 2:0}
	for i in range(no_games):
		board, game_result = connectfour.PlayGame(user_type_in, model_in, play_start)
		outcome_tot[game_result]+=1
	return outcome_tot

def ObjFun(weights_vec):
	# set model weights
	weights_vec = np.array(weights_vec)
	weights_norm = np.multiply(weights_vec,norm_arr)
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
	view board which lights up each layer most
	try algorithm/hall of fame from DEAP
	GA early stopping - max score or no changes
	""")

# Define neural network
layer_params=[['dense', 20 , 'sigmoid'],['dense', 10 , 'sigmoid'],['dense', 7, 'softmax']]
model_untrained = NeuralNetwork(layer_params)
weights_shape = [mat.shape for mat in model_untrained.get_weights()]
weights_size = [mat.size for mat in model_untrained.get_weights()]
norm_list=[]
for i in range(int(len(weights_size)/2)):
	norm_list.append(np.ones((sum(weights_size[2*i:2*(i+1)])))*2/(weights_size[2*i])**0.5)
norm_arr = np.concatenate(norm_list)

# Define game parameters
types_in = {1: 'NeuralNetwork', 2:'Rand'}
no_games = 10  

# Define optimisaion parameters
pop_size = 32
mut_prob = 0.2
max_gen = 1000
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

ev_best = floatopt(toolbox, tools, pop_size, mut_prob, max_gen)