import cargame
import neat
import os
import time
import visualize
import pickle

print("""
To do:
	deep Q learning - install (custom) tf and tf agent for py3.5
	""")

def PickleSave(item, filename):
	with open(filename, 'wb') as fp:
		pickle.dump(item, fp)

def PickleLoad(filename):
	with open(filename, 'rb') as fp:
		out = pickle.load(fp)
	return out

def EvalGenomes(genomes, config):
	networks = []
	for __, genome in genomes:
		networks.append(neat.nn.FeedForwardNetwork.create(genome, config))
	scores = cargame.PlayGame(networks, 500, 2, 2)
	for (__, genome), score in zip(genomes, scores):
		genome.fitness = score

def AddStats(p_in):
	p_in.add_reporter(neat.StdOutReporter(True))
	p_in.add_reporter(stats)
	p_in.add_reporter(checkpointer)
	return p_in

# cargame.PlayGame([None], 1000, 1, 30)

# Initialise optimisation
stats = neat.StatisticsReporter()
checkpointer = neat.Checkpointer(10)
try:
	p = neat.Checkpointer.restore_checkpoint('neat-checkpoint')
except:
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'neat_config')
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
		neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)
	p = neat.Population(config)

# Run optimisation
p = AddStats(p)
winner = p.run(EvalGenomes, 2)

# Visualise results
visualize.plot_stats(stats, ylog=False, view=False)
visualize.draw_net(config, winner, False, filename='architecture', fmt = 'png')

# Save best genome
filename = 'best_genome'
PickleSave(winner, filename)

# Load best genome and play game
best_genome = PickleLoad(filename)
best_net = [neat.nn.FeedForwardNetwork.create(best_genome, config)]
cargame.PlayGame(best_net, None, 1000, None)