import cargame
import neat
import os
import time
import visualize

def EvalGenomes(genomes, config):
	networks = []
	for __, genome in genomes:
		networks.append(neat.nn.FeedForwardNetwork.create(genome, config))
	scores = cargame.PlayGame(networks, 2000, 1, 10)
	# print(scores)
	for (__, genome), score in zip(genomes, scores):
		genome.fitness = score
	# time.sleep(10)

def AddStats(p_in):
	p_in.add_reporter(neat.StdOutReporter(True))
	p_in.add_reporter(stats)
	p_in.add_reporter(checkpointer)
	return p_in

# Load configuration
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'neat_config incomplete')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
	neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)

# Initialise
stats = neat.StatisticsReporter()
checkpointer = neat.Checkpointer(1)

try:
	p = neat.Checkpointer.restore_checkpoint('neat-checkpoint')
except:
	p = neat.Population(config)
p = AddStats(p)
winner = p.run(EvalGenomes, 10000)
print('Perfected by generation {1}'.format(checkpointer.last_generation_checkpoint))


p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-{0}'.format(checkpointer.last_generation_checkpoint))
p = AddStats(p)
winner = p.run(EvalGenomes, 10000)
print('Perfected by generation {1}'.format(checkpointer.last_generation_checkpoint))


# Display the progress after each generation
visualize.plot_stats(stats, ylog=False, view=False)
visualize.plot_species(stats, view=False)