import cargame
import neat
import os
import time
import visualize

print("""
To do:
	add convex polygon
	deep Q learning - install (custom) tf and tf agent for py3.5
	""")

def EvalGenomes(genomes, config):
	networks = []
	for __, genome in genomes:
		networks.append(neat.nn.FeedForwardNetwork.create(genome, config))
	scores = cargame.PlayGame(networks, 100, 2, 35)
	for (__, genome), score in zip(genomes, scores):
		genome.fitness = score

def AddStats(p_in):
	p_in.add_reporter(neat.StdOutReporter(True))
	p_in.add_reporter(stats)
	p_in.add_reporter(checkpointer)
	return p_in

# cargame.PlayGame([None], 1000, 1, 30)

# Initialise
stats = neat.StatisticsReporter()
checkpointer = neat.Checkpointer(10)

try:
	p = neat.Checkpointer.restore_checkpoint('neat-checkpoint')
except:
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'neat_config incomplete')
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
		neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)
	p = neat.Population(config)
p = AddStats(p)
winner = p.run(EvalGenomes, 100)
print('Perfected by generation {0}'.format(checkpointer.last_generation_checkpoint))
visualize.plot_stats(stats, ylog=False, view=False)
visualize.draw_net(config, winner, False, filename='architecture', fmt = 'png')

# p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-{0}'.format(checkpointer.last_generation_checkpoint))
# p = AddStats(p)
# winner = p.run(EvalGenomes, 10)
# print('Perfected by generation {0}'.format(checkpointer.last_generation_checkpoint))
# visualize.plot_stats(stats, ylog=False, view=True)