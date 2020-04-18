# Import libraries
import random
random.seed(42)

# Define function
def floatopt(toolbox, tools, pop_sz, MUTPB, g_max, fit_max):
    # initialise
    g = 0
    pop = toolbox.population(n=pop_sz)
    # evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # print the stats
    fits = [ind.fitness.values[0] for ind in pop]
    print(f"Evolution {g} : Min {min(fits):.3f}, Max {max(fits):.3f}, Mean {sum(fits) / len(pop):.3f}")
    best_ind = [tools.selBest(pop, 1)[0]]
    while g < g_max and max(fits)<fit_max:
        g+=1
        # select and clone the next generation individuals
        unchanged = toolbox.select(pop, int(len(pop)/4))
        unchanged = list(map(toolbox.clone, unchanged))
        # apply crossover and mutation
        mated = []
        for i in range(int((len(pop)-len(unchanged))/2)):
            child1, child2 = list(map(toolbox.clone, random.sample(pop,2)))
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
            mated.append(child1)
            mated.append(child2)
        pop[:] = unchanged + mated
        for mutant in pop:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # save the best
        best_ind.append(tools.selBest(pop, 1)[0])
        # print the stats
        if g % 1 == 0:
            fits = [ind.fitness.values[0] for ind in pop]
            print(f"Evolution {g:2.0f} : Min {min(fits):.3f}, Max {max(fits):.3f}, Mean {sum(fits) / len(pop):.3f}")
    return best_ind