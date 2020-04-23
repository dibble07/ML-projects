# from __future__ import print_function
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
import connectfour
import neat
import os
import visualize

def ManyGames(user_type_in, model_in, play_start):
	outcome_tot={0:0, 1:0, 2:0, -1:0, -2:0}
	for i in range(np.floor(no_games/2)):
		__, __, game_result = connectfour.PlayGame(user_type_in, model_in, play_start)
		outcome_tot[game_result]+=1
	return outcome_tot

def ObjFun(model_in):
	outcomes_all = {0:0, 1:0, 2:0, -1:0, -2:0}
	# player 1 goes first
	outcomes = ManyGames(types_in, model_in, 1)
	outcomes_all[0]+=outcomes[0]
	outcomes_all[1]+=outcomes[1]
	outcomes_all[2]+=outcomes[2]
	outcomes_all[-1]+=outcomes[-1]
	outcomes_all[-2]+=outcomes[-2]
	# player 2 goes first
	outcomes = ManyGames(types_in, model_in, 2)
	outcomes_all[0]+=outcomes[0]
	outcomes_all[1]+=outcomes[1]
	outcomes_all[2]+=outcomes[2]
	outcomes_all[-1]+=outcomes[-1]
	outcomes_all[-2]+=outcomes[-2]
	# calulate score
	if obj == "legal moves":
		score_out = -outcomes_all[-1]/no_games
	elif obj == "winning and not losing":
		score_out = ((outcomes_all[1]-outcomes_all[-1]) - (outcomes_all[2]-outcomes_all[-2]))/no_games-1
	# print(outcomes_all, score)
	return score_out

def EvalGenomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = ObjFun(neat.nn.FeedForwardNetwork.create(genome, config))

def AddStats(p_in):
	p_in.add_reporter(neat.StdOutReporter(True))
	p_in.add_reporter(stats)
	p_in.add_reporter(checkpointer)
	return p_in

# Load configuration
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'neat_config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
	neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)

# Test games
# types_in = {1:'Rand', 2:'Rand'}
# connectfour.PlayGame(types_in,[],1)
# print(ObjFun([]))

# Initialise
stats = neat.StatisticsReporter()
checkpointer = neat.Checkpointer(1)
types_in = {1:'NeuralNetwork', 2:'Rand'}


try:
	p = neat.Checkpointer.restore_checkpoint('neat-checkpoint')
except:
	p = neat.Population(config)
p = AddStats(p)
no_games = 50
obj = "legal moves"
winner = p.run(EvalGenomes, 10000)
print('Perfected at {0} game level by generation {1}'.format(no_games, checkpointer.last_generation_checkpoint))


p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-{0}'.format(checkpointer.last_generation_checkpoint))
p = AddStats(p)
no_games = 100
obj = "legal moves"
winner = p.run(EvalGenomes, 10000)
print('Perfected at {0} game level by generation {1}'.format(no_games, checkpointer.last_generation_checkpoint))


p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-{0}'.format(checkpointer.last_generation_checkpoint))
p = AddStats(p)
no_games = 1000
obj = "legal moves"
winner = p.run(EvalGenomes, 10000)
print('Perfected at {0} game level by generation {1}'.format(no_games, checkpointer.last_generation_checkpoint))


p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-{0}'.format(checkpointer.last_generation_checkpoint))
p = AddStats(p)
no_games = 50
obj = "winning and not losing"
winner = p.run(EvalGenomes, 10000)
print('Perfected at {0} game level by generation {1}'.format(no_games, checkpointer.last_generation_checkpoint))


# Display the progress after each generation
visualize.plot_stats(stats, ylog=False, view=False)
visualize.plot_species(stats, view=False)