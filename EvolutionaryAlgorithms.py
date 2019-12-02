import random
import numpy as np
import FeedforwardNetwork as ffn
import MathAndStats as ms


def generateFFNGA(mlp, training_set, testing_set, nodes_by_layer,
                  prob_cross, prob_mutation, mutation_variance, max_generations, population_size):
    population = initializePopulationFFN(mlp.out_k, nodes_by_layer, population_size)
    #mlp = ffn.FeedforwardNetwork(out_k, clsses, output_type, logistic_nodes, logistic_output)
    population_fitness = evaluateGroupFitnessFFN(mlp, testing_set, nodes_by_layer, population, mlp.output_type)
    #mlp_1.setWeights(nodes_by_layer, weights)
    for generation in range(1, max_generations):
        print("Working on generation", generation,"of",max_generations)
        selection_group = selectChromosomesGA(population, population_fitness, k=2)
        recombined_group = recombineGA(selection_group, prob_cross, population_size)
        mutateGA(recombined_group, prob_mutation, mutation_variance)
        population_fitness = evaluateGroupFitnessFFN(mlp, testing_set, nodes_by_layer, recombined_group, mlp.output_type)
        population = recombined_group
    best_fitness_index = 0
    for index in range(1, len(population)):
        if population_fitness[index] > population_fitness[best_fitness_index]:
            best_fitness_index = index
    most_fit_chromosome = population[best_fitness_index]
    mlp.setWeights(nodes_by_layer, most_fit_chromosome)
    #return mlp


# initialize a population of chromosomes with random weights in [-0.3,0.3]
# along a uniform distribution
def initializePopulationFFN(out_k, nodes_by_layer, population_size):
    # find length of chromosome
    num_weights = 0
    for layer in range(1, len(nodes_by_layer)):
        num_weights += (nodes_by_layer[layer] * (nodes_by_layer[layer - 1] + 1))
    num_weights += out_k * (nodes_by_layer[-1] + 1)

    # initialize population chromosomes
    population = []
    # generate chromosomes in the population
    for chromosome in range(population_size):
        chromo = []
        for weight_index in range(num_weights):
            chromo.append(random.uniform(-0.3, 0.3))
        population.append(chromo)
    return population

# evaluate the fitness of a group of chromosomes
# fitness has been chosen to be the the negative of their typical error (MSE/0-1 loss/etc)
def evaluateGroupFitnessFFN(mlp, testing_set, nodes_by_layer, group, output_type):
    group_fitness = []
    for chromosome_num in range(len(group)):
        # assign chromosome
        mlp.setWeights(nodes_by_layer, group[chromosome_num])
        # classification
        if output_type == "classification":
            results = ms.testClassifier(mlp, testing_set)
            fitness = 0
            for run in range(len(results)):
                fitness += results[run]
            fitness *= -1 / len(results)
            group_fitness.append(fitness)
        # autoencoder
        elif output_type == "autoencoder":
            group_fitness.append(-1 * ms.testAutoencoder(mlp, testing_set))
        # regression
        else:
            # get error for test
            results = ms.testRegressor(mlp, testing_set)
            fitness = 0
            for obs in range(len(results)):
                fitness += (results[obs] * results[obs])
            fitness *= -1 / len(results)
            group_fitness.append(fitness)
        # clear network
        mlp.hidden_layers = []
        mlp.output_layer = []
    return group_fitness

# choose selection group of size 1/4 of the original population size using k tournament selection
def selectChromosomesGA(population, population_fitness, k):
    selection_group = []
    # for loop iterating for the length of the desired selection group
    for pair in range(int(len(population)/4)):
        tournament_chromosomes = []
        tournament_chromosomes_fitnesses = []
        # get k random chromosomes from the population
        for i in range(k):
            selected_index = random.randint(0, len(population)-1)
            tournament_chromosomes.append(population[selected_index])
            tournament_chromosomes_fitnesses.append(population_fitness[selected_index])
        # select the chromosome with the highest fitness and add it to the selection group
        lowest_index = 0
        for index in range(1, k):
            if tournament_chromosomes_fitnesses[index] > tournament_chromosomes_fitnesses[lowest_index]:
                lowest_index = index
        selection_group.append(tournament_chromosomes[lowest_index])

    return selection_group

# breed chromosomes using 2 parents: 2 children using prob_cross for probability of crossover
def recombineGA(selection_group, prob_cross, population_size):
    recombined_group = []
    # continue recombining until the recombined population is the same size as the original population
    while len(recombined_group) < population_size:
        # get 2 random parents
        parents = []
        parents.append(selection_group[random.randint(0, len(selection_group)-1)])
        parents.append(selection_group[random.randint(0, len(selection_group)-1)])
        # if we do crossover
        if random.uniform(0, 1) < prob_cross:
            children = []
            children.append([])
            children.append([])
            for gene in range(len(parents[0])):
                # if we take gene from parent 0
                if random.uniform(0, 1) < 0.5:
                    children[0].append(parents[0][gene])
                    children[1].append(parents[1][gene])
                else:
                    children[0].append(parents[1][gene])
                    children[1].append(parents[0][gene])
            recombined_group.append(children[0])
            recombined_group.append(children[1])
        # else add parents directly
        else:
            recombined_group.append(parents[0])
            recombined_group.append(parents[1])
    # if too many chromosomes were added, remove extras
    while len(recombined_group) > population_size:
        del recombined_group[random.randint(0, len(recombined_group)-1)]
    return recombined_group

# mutate chromosomes of the next generation
def mutateGA(recombined_group, prob_mutation, mutation_variance):
    for chromosome in range(len(recombined_group)):
        # if we mutate, add Gaussian noise to all genes
        if random.uniform(0, 1) < prob_mutation:
            # get Gaussian noise with mean 0 and variance mutation_variance
            noise = np.random.normal(0, mutation_variance, len(recombined_group[chromosome]))
            for gene in range(len(recombined_group[chromosome])):
                recombined_group[chromosome][gene] += noise[gene]