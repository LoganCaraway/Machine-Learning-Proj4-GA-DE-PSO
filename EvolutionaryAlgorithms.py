import random
import copy
import MathAndStats as ms

# implementation of the genetic algorithm to tune the weights of a feedforward network
def generateFFNGA(mlp, training_set, validation_set, nodes_by_layer,
                  prob_cross, prob_mutation, mutation_variance, population_size, max_generations):
    population = initializePopulationFFN(mlp.out_k, nodes_by_layer, population_size, -1, 1)
    population_fitness = evaluateGroupFitnessFFN(mlp, training_set, nodes_by_layer, population, mlp.output_type)
    for generation in range(1, max_generations):
        print("GA: Working on generation", generation,"of",max_generations,"for",len(nodes_by_layer)-1,"layer MLP")
        print("Using: probability of crossing=",prob_cross,"probability of mutation=",prob_mutation, "mutation variance",
              mutation_variance,"population size=",population_size, "max generations=",max_generations)
        selection_group = selectChromosomesGA(population, population_fitness, k=2, num_winning_pairs=2)
        recombined_group = recombineGA(selection_group, prob_cross)
        mutateGA(recombined_group, prob_mutation, mutation_variance)
        #population = replace(population, recombined_group, population_fitness)
        # get fitness for new chromosomes (recombined_group)
        recombined_fitness = evaluateGroupFitnessFFN(mlp, training_set, nodes_by_layer, recombined_group, mlp.output_type)
        population, population_fitness = replace(population, recombined_group, population_fitness, recombined_fitness)
        # population = replace(population, recombined_group)
        average_fitness = ms.getMean(population_fitness, len(population_fitness))
        print("GA: Average fitness for the generation: ", average_fitness)
    # evaluate the final population using the validation set to help fight overfitting
    population_fitness = evaluateGroupFitnessFFN(mlp, validation_set, nodes_by_layer, population, mlp.output_type)
    best_fitness_index = 0
    for index in range(1, len(population)):
        if population_fitness[index] > population_fitness[best_fitness_index]:
            best_fitness_index = index
    most_fit_chromosome = population[best_fitness_index]
    mlp.setWeights(nodes_by_layer, most_fit_chromosome)


# initialize a population of chromosomes with random weights in [min, max]
# along a uniform distribution
def initializePopulationFFN(out_k, nodes_by_layer, population_size, min, max):
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
            chromo.append(random.uniform(min, max))
            #chromo.append(random.gauss(0, max))
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
            results = ms.testProbabilisticClassifier(mlp, testing_set)
            fitness = 0
            for run in range(len(results)):
                fitness += results[run][0]
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

# choose selection group of size 1/2 of the original population size using k tournament selection
def selectChromosomesGA(population, population_fitness, k, num_winning_pairs):
    selection_group = []
    # for loop iterating for the length of the desired selection group
    for pair in range(num_winning_pairs * 2):
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
        selection_group.append(copy.deepcopy(tournament_chromosomes[lowest_index]))

    return selection_group

# breed chromosomes using 2 parents: 2 children using prob_cross for probability of crossover
def recombineGA(selection_group, prob_cross):
    recombined_group = []
    #while len(recombined_group) < len(selection_group):
    for pair in range(0, len(selection_group), 2):
        # get parents
        parents = []
        parents.append(selection_group[pair])
        parents.append(selection_group[pair+1])
        # if we do crossover
        if random.uniform(0, 1) <= prob_cross:
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
    return recombined_group

# mutate chromosomes of the next generation
def mutateGA(recombined_group, prob_mutation, mutation_variance):
    for chromosome in range(len(recombined_group)):
        # if we mutate, add Gaussian noise to all genes
        if random.uniform(0, 1) <= prob_mutation:
            # get Gaussian noise with mean 0 and variance mutation_variance
            for gene in range(len(recombined_group[chromosome])):
                recombined_group[chromosome][gene] += random.gauss(0, mutation_variance)

# generate the next generation
# add the most fit to the recombined group, then add from the previous population with replacement randomly
def replace(population, recombined_group, population_fitness, recombined_fitness):
    # add most fit member to next generation
    best_fitness_index = 0
    for index in range(1, len(population)):
        if population_fitness[index] > population_fitness[best_fitness_index]:
            best_fitness_index = index
    recombined_group.append(population[best_fitness_index])
    recombined_fitness.append(population_fitness[best_fitness_index])
    print("Best fitness from previous population: ", recombined_fitness[-1])
    # add the rest of the previous generation randomly to fill out the next generation
    while len(recombined_group) < len(population):
        selected_index = random.randint(0, len(population) - 1)
        recombined_group.append(population[selected_index])
        recombined_fitness.append(population_fitness[selected_index])
    return recombined_group, recombined_fitness

# implementation of DE/rand/1/bin for tuning a feedforward network
def generateFFNDE(mlp, training_set, validation_set, nodes_by_layer,
                  prob_target, beta, population_size, max_generations):
    population = initializePopulationFFN(mlp.out_k, nodes_by_layer, population_size, -1, 1)
    population_fitness = evaluateGroupFitnessFFN(mlp, training_set, nodes_by_layer, population, mlp.output_type)
    for generation in range(1, max_generations):
        print("\n\nDE: Working on generation", generation, "of", max_generations, "for", len(nodes_by_layer) - 1,
              "layer MLP")
        print("Using: probability of using target gene=", prob_target, "beta=", beta,
              "population size=", population_size, "max generations=", max_generations)
        next_generation = []
        next_generation_fitness = []
        for target_vector_index in range(population_size):
            target_vector = population[target_vector_index]
            target_fitness = population_fitness[target_vector_index]
            # select a random vector as the target vector
            #target_index = random.randint(0, len(population)-1)
            #target_vector = population[target_index]
            #target_fitness = population_fitness[target_index]
            trial_vector = mutateDE(population, population_fitness, beta)
            offspring = crossoverDE(target_vector, trial_vector, prob_target)
            offspring_fitness = evaluateGroupFitnessFFN(mlp, training_set, nodes_by_layer, [trial_vector], mlp.output_type)[0]
            if offspring_fitness >= target_fitness:
                next_generation.append(offspring)
                next_generation_fitness.append(offspring_fitness)
                print("Fitness from offspring: ", offspring_fitness)
            else:
                next_generation.append(target_vector)
                next_generation_fitness.append(target_fitness)
                print("Fitness from target: ", target_fitness)
        population = next_generation
        population_fitness = next_generation_fitness
        average_fitness = ms.getMean(population_fitness, len(population_fitness))
        print("\n\nDE: Average fitness for the generation: ", average_fitness)
        best_fitness_index = 0
        for index in range(1, len(population)):
            if population_fitness[index] > population_fitness[best_fitness_index]:
                best_fitness_index = index
        print("Best fitness for generation:",population_fitness[best_fitness_index])
    # evaluate the final population using the validation set to help fight overfitting
    population_fitness = evaluateGroupFitnessFFN(mlp, validation_set, nodes_by_layer, population, mlp.output_type)
    # return most fit chromosome
    best_fitness_index = 0
    for index in range(1, len(population)):
        if population_fitness[index] > population_fitness[best_fitness_index]:
            best_fitness_index = index
    most_fit_chromosome = population[best_fitness_index]
    mlp.setWeights(nodes_by_layer, most_fit_chromosome)

# implementes mutation with 1 difference vector for differential evolution
def mutateDE(population, population_fitness, beta):
    #trial_vector = population[random.randint(0, len(population)-1)]
    best_fitness_index = 0
    for index in range(1, len(population)):
        if population_fitness[index] > population_fitness[best_fitness_index]:
            best_fitness_index = index
    trial_vector = copy.deepcopy(population[best_fitness_index])
    for gene in range(len(trial_vector)):
        difference = beta * ( population[random.randint(0, len(population)-1)][gene] -
                                       population[random.randint(0, len(population)-1)][gene] )
        if trial_vector[gene] + difference > 1.5:
            trial_vector[gene] = 3 - ( trial_vector[gene] + difference )
        elif trial_vector[gene] + difference < -1.5:
            trial_vector[gene] = -3 + ( trial_vector[gene] + difference )
        else:
            trial_vector[gene] += difference
    #trial_vector += beta * ( population[random.randint(0, len(population)-1)] - population[random.randint(0, len(population)-1)])
    return trial_vector

# implements uniform crossover for differential evolution
def crossoverDE(target_vector, trial_vector, prob_target):
    new_vector = []
    for gene in range(len(target_vector)):
        if random.uniform(0, 1) <= prob_target:
            new_vector.append(target_vector[gene])
        else:
            new_vector.append(trial_vector[gene])
    return new_vector