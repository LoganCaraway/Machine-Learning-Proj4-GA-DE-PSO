import random
import copy
import MathAndStats as ms

def generateFFNgBestPSO(mlp, training_set, validation_set, nodes_by_layer, omega, c1, c2,
                        population_size, max_time_steps):
    population = initializePopulationFFN(mlp.out_k, nodes_by_layer, population_size, -1, 1)
    population_velocities = initializeVelocities(population, -1, 1)
    population_fitness = evaluateGroupFitnessFFN(mlp, training_set, nodes_by_layer, population, mlp.output_type)
    pbest_positions = copy.deepcopy(population)
    pbest_fitnesses = copy.deepcopy(population_fitness)
    # get global best
    best_fitness_index = 0
    for index in range(1, len(population)):
        if population_fitness[index] > population_fitness[best_fitness_index]:
            best_fitness_index = index
    gbest_position = copy.deepcopy(population[best_fitness_index])
    gbest_fitness = population_fitness[best_fitness_index]
    print("PSO Initial global best fitness:", gbest_fitness)
    for time_step in range(max_time_steps):
        print("\n\nPSO Time step",time_step,"of",max_time_steps)
        print("Using omega=",omega,"c1=",c1,"c2=",c2,"population size=",population_size)
        population_fitness = evaluateGroupFitnessFFN(mlp, training_set, nodes_by_layer, population, mlp.output_type)
        print("PSO: Average fitness for current time step:",ms.getMean(population_fitness, population_size))
        # update personal and global bests
        for particle in range(population_size):
            # set the personal best position and fitness
            if (population_fitness[particle] > pbest_fitnesses[particle]):
                # update personal best to current state
                pbest_positions[particle] = copy.deepcopy(population[particle])
                pbest_fitnesses[particle] = copy.deepcopy(population_fitness[particle])
            # set the global best position and fitness
            if pbest_fitnesses[particle] > gbest_fitness:
                # set global best to current location
                gbest_position = pbest_positions[particle]
                gbest_fitness = pbest_fitnesses[particle]
                print("New global best fitness:", gbest_fitness)
        # update velocity and position
        for particle in range(population_size):
            for element in range(len(population[particle])):
                phi1 = c1 * random.uniform(0, 1)
                phi2 = c2 * random.uniform(0, 1)
                population_velocities[particle][element] = omega * population_velocities[particle][element] + (
                        phi1 * (pbest_positions[particle][element] - population[particle][element])) + (
                        phi2 * (gbest_position[element] - population[particle][element]))
                # if the particle hits the edge of the search space, bounce off
                # and move back the amount out of bound the particle would have been
                if population[particle][element] + population_velocities[particle][element] > 1.5:
                    population[particle][element] = 3 - (population[particle][element] + population_velocities[particle][element])
                    #population_velocities[particle][element] = 0
                elif population[particle][element] + population_velocities[particle][element] < -1.5:
                    population[particle][element] = -3 + (population[particle][element] + population_velocities[particle][element])
                    #population_velocities[particle][element] = 0
                else:
                    population[particle][element] += population_velocities[particle][element]
    # evaluate the final population and the gbest using the validation set to help fight overfitting
    # include gbest in the final selection group
    population.append(gbest_position)
    population_fitness = evaluateGroupFitnessFFN(mlp, validation_set, nodes_by_layer, population, mlp.output_type)
    best_fitness_index = 0
    for index in range(1, len(population)):
        if population_fitness[index] > population_fitness[best_fitness_index]:
            best_fitness_index = index
    most_fit_chromosome = population[best_fitness_index]
    mlp.setWeights(nodes_by_layer, most_fit_chromosome)


# initialize a population of vectors with random weights in [min, max]
# along a uniform distribution
def initializePopulationFFN(out_k, nodes_by_layer, population_size, min, max):
    # find length of vectors
    num_weights = 0
    for layer in range(1, len(nodes_by_layer)):
        num_weights += (nodes_by_layer[layer] * (nodes_by_layer[layer - 1] + 1))
    num_weights += out_k * (nodes_by_layer[-1] + 1)

    # initialize population vectors
    population = []
    # generate vectors in the population
    for vector_num in range(population_size):
        vector = []
        for weight_index in range(num_weights):
            vector.append(random.uniform(min, max))
        population.append(vector)
    return population

# initialize the velocity of the vectors
def initializeVelocities(population, state_min, state_max):
    population_velocities = []
    for vector in range(len(population)):
        vector_velocity = []
        for element in range(len(population[vector])):
            vector_velocity.append(random.uniform(-1 * (state_max - state_min) / 10, (state_max - state_min) / 10))
        population_velocities.append(vector_velocity)
    return  population_velocities

# evaluate the fitness of a group of chromosomes
# fitness has been chosen to be the the negative of their typical error (MSE/0-1 loss/etc)
def evaluateGroupFitnessFFN(mlp, testing_set, nodes_by_layer, group, output_type):
    group_fitness = []
    for vector_num in range(len(group)):
        # assign vector
        mlp.setWeights(nodes_by_layer, group[vector_num])
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
