# Import libraries
print("Importing libraries")
from geneticoptimiser import floatopt
from deap import base
from deap import creator
from deap import tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import random
random.seed(42)

# User defined functions
def Rosenbrock(vec_in):
    a=3
    b=100
    x=vec_in[0]
    y=vec_in[1]
    rosenbrock = (a-x)**2+b*(y-x**2)**2
    return rosenbrock,

# Define optimisaion parameters
pop_size = 64
# cross_prob = 0.2
mut_prob = 0.5
max_gen = 100000
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", Rosenbrock)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.5)
toolbox.register("select", tools.selBest)

ev_best = floatopt(toolbox, tools, pop_size, mut_prob, max_gen)
# print(ev_best['fit'])
# print(ev_best)

hey = []
hello = []
for i, pop in enumerate(ev_best[::10]):
    hey.append(np.mean(np.array(pop), axis=0))
    hi = [ind.fitness.values[0] for ind in pop]
    hello.append(pop[np.argmax(hi)])
ho=np.array(hey)
hip=np.array(hello)
plt.plot(hip[:,0], hip[:,1], label = 'Best')
plt.plot(ho[:,0], ho[:,1], label = 'Avg')
plt.legend()
plt.show()