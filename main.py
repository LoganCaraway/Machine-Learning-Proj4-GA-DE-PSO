import MathAndStats as ms
import csv
import sys
import random
import math
#import NearestNeighbor as nn
#import Kmeans as km
#import PAM as pam
#import RBFNetwork as rbf
import FeedforwardNetwork as ffn
import EvolutionaryAlgorithms as ea
import PSO

#--------------------DATA-MANIPULATION--------------------#
def openFiles(dataFile):
    lines = open(dataFile, "r").readlines()
    csvLines = csv.reader(lines)
    data = list()
    #save = None

    for line in csvLines:
        tmp = []
        for c in range(0, len(line) - 1):
            tmp.append(float(line[c]))
        if sys.argv[2] == 'r':
            tmp.append(float(line[-1]))
        else:
            tmp.append(line[-1])
        data.append(tmp)

    # remove line number from each example (first column)
    for example in range(len(data)):
        del data[example][0]

    data = ms.normalize(data)
    #print(data)
    if sys.argv[1] == "output_machine.data":
        for obs in range(len(data)):
            if int(data[obs][-1]) < 21:
                data[obs][-1] = 1
            elif int(data[obs][-1]) < 101:
                data[obs][-1] = 2
            elif int(data[obs][-1]) < 201:
                data[obs][-1] = 3
            elif int(data[obs][-1]) < 301:
                data[obs][-1] = 4
            elif int(data[obs][-1]) < 401:
                data[obs][-1] = 5
            elif int(data[obs][-1]) < 501:
                data[obs][-1] = 6
            elif int(data[obs][-1]) < 601:
                data[obs][-1] = 7
            else:
                data[obs][-1] = 8

    if (len(sys.argv) > 4) and (sys.argv[4] == "log"):
        logOutputs(data)
    # divide data into 10 chunks for use in 10-fold cross validation paired t test
    chnks = getNChunks(data, 10)
    class_list = getClasses(data)

    # get a boolean vector telling whether to use euclidean distance or hamming distance on a feature-by-feature basis
    #data_metric = getDataMetrics()

    return chnks, class_list

# divide the example set into n random chunks of approximately equal size
def getNChunks(data, n):
    # randomly shuffle the order of examples in the data set
    random.shuffle(data)
    dataLen = len(data)
    chunkLen = int(dataLen / n)
    # chunks is a list of the individual chunks
    chunks = []
    # rows are observation
    # columns are labels

    # skip along the data file chunking every chunkLen
    for index in range(0, dataLen, chunkLen):
        if (index + chunkLen) <= dataLen:
            # copy from current skip to the next
            chunk = data[index:index + chunkLen]
            # chunks is a list of the individual chunks
            chunks.append(chunk)
    # append the extra examples to the last chunk
    for i in range(n*chunkLen, dataLen):
        chunks[-1].append(data[i])
    for i in range(len(chunks)):
        print("Length of chunk: ", len(chunks[i]))
    return chunks
#--------------------DATA-MANIPULATION-END--------------------#

def logOutputs(data):
    if not sys.argv[2] == 'r':
        print("Only log outputs for regression")
        return
    for example in range(len(data)):
        temp = data[example][-1]
        del data[example][-1]
        if temp == 0:
            temp = 0.001
        data[example].append(math.log(temp))

def getClasses(data):
    if sys.argv[2] == 'r':
        return []
    classes = []
    for x in range(len(data)):
        if not data[x][-1] in classes:
            classes.append(data[x][-1])
    return classes

def trainAndTest(chunked_data, clss_list, k, use_regression, num_layers, hidden_layer_nodes, eta, alpha_momentum, iterations, tune):
    mlp_BP_0_missed = []
    mlp_BP_1_missed = []
    mlp_BP_2_missed = []
    mlp_GA_0_missed = []
    mlp_GA_1_missed = []
    mlp_GA_2_missed = []
    mlp_DE_0_missed = []
    mlp_DE_1_missed = []
    mlp_DE_2_missed = []
    mlp_PSO_0_missed = []
    mlp_PSO_1_missed = []
    mlp_PSO_2_missed = []
    for testing in range(10):
        print("\n\n\nFold: ",testing,"of 10 fold cross validation")
        training_set = []

        testing_set = chunked_data[testing]
        # make example set
        for train in range(10):
            if train != testing:
                for x in range(len(chunked_data[train])):
                    training_set.append(chunked_data[train][x])

        validation_index = int((float(len(training_set)) * 8 / 10)) - 1
        if use_regression:
            # 0-layer
            mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            mlp.backpropogation(training_set, hidden_layer_nodes[:0], eta, alpha_momentum, iterations)
            mlp_BP_0_missed.append(ms.testRegressor(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            ea.generateFFNGA(mlp, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:0],
                             prob_cross=0.6, prob_mutation=0.01, mutation_variance=0.2, population_size=40,
                             max_generations=1000)
            mlp_GA_0_missed.append(ms.testRegressor(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            ea.generateFFNDE(mlp, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:0],
                             prob_target=0.5, beta=0.5, population_size=40, max_generations=1000)
            mlp_DE_0_missed.append(ms.testRegressor(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            PSO.generateFFNgBestPSO(mlp, training_set[:validation_index], training_set[validation_index:],
                                    [len(chunks[0][0]) - 1] + hidden_layer_nodes[:0],
                                    omega=0.8, c1=0.6, c2=0.6, population_size=40, max_time_steps=8000)
            mlp_PSO_0_missed.append(ms.testRegressor(mlp, testing_set))
            # 1-layer
            mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            mlp.backpropogation(training_set, hidden_layer_nodes[:1], eta, alpha_momentum, iterations)
            mlp_BP_1_missed.append(ms.testRegressor(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            ea.generateFFNGA(mlp, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:1],
                             prob_cross=0.6, prob_mutation=0.01, mutation_variance=0.2, population_size=40,
                             max_generations=300)
            mlp_GA_1_missed.append(ms.testRegressor(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            ea.generateFFNDE(mlp, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:1],
                             prob_target=0.5, beta=0.5, population_size=40, max_generations=100)
            mlp_DE_1_missed.append(ms.testRegressor(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            PSO.generateFFNgBestPSO(mlp, training_set[:validation_index], training_set[validation_index:],
                                    [len(chunks[0][0]) - 1] + hidden_layer_nodes[:1],
                                    omega=0.8, c1=0.6, c2=0.6, population_size=40, max_time_steps=80)
            mlp_PSO_1_missed.append(ms.testRegressor(mlp, testing_set))
            # 2-layer
            mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            mlp.backpropogation(training_set, hidden_layer_nodes[:2], eta, alpha_momentum, iterations)
            mlp_BP_2_missed.append(ms.testRegressor(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            ea.generateFFNGA(mlp, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:2],
                             prob_cross=0.6, prob_mutation=0.01, mutation_variance=0.2, population_size=40,
                             max_generations=300)
            mlp_GA_2_missed.append(ms.testRegressor(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            ea.generateFFNDE(mlp, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:2],
                             prob_target=0.5, beta=0.5, population_size=40, max_generations=100)
            mlp_DE_2_missed.append(ms.testRegressor(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            PSO.generateFFNgBestPSO(mlp, training_set[:validation_index], training_set[validation_index:],
                                    [len(chunks[0][0]) - 1] + hidden_layer_nodes[:2],
                                    omega=0.8, c1=0.6, c2=0.6, population_size=40, max_time_steps=80)
            mlp_PSO_2_missed.append(ms.testRegressor(mlp, testing_set))

        # classification
        else:
            # 0-layer
            mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            mlp.backpropogation(training_set, hidden_layer_nodes[:0], eta, alpha_momentum, iterations)
            mlp_BP_0_missed.append(ms.testProbabilisticClassifier(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            ea.generateFFNGA(mlp, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:0],
                             prob_cross=0.6, prob_mutation=0.01, mutation_variance=0.2, population_size=60,
                             max_generations=300)
            mlp_GA_0_missed.append(ms.testProbabilisticClassifier(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            ea.generateFFNDE(mlp, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:0],
                             prob_target=0.5, beta=0.5, population_size=40, max_generations=100)
            mlp_DE_0_missed.append(ms.testProbabilisticClassifier(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            PSO.generateFFNgBestPSO(mlp, training_set[:validation_index], training_set[validation_index:],
                                    [len(chunks[0][0]) - 1] + hidden_layer_nodes[:0],
                                    omega=0.8, c1=0.6, c2=0.6, population_size=50, max_time_steps=80)
            mlp_PSO_0_missed.append(ms.testProbabilisticClassifier(mlp, testing_set))
            # 1-layer
            mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            mlp.backpropogation(training_set, hidden_layer_nodes[:1], eta, alpha_momentum, iterations)
            mlp_BP_1_missed.append(ms.testProbabilisticClassifier(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            ea.generateFFNGA(mlp, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:1],
                             prob_cross=0.6, prob_mutation=0.01, mutation_variance=0.2, population_size=60,
                             max_generations=300)
            mlp_GA_1_missed.append(ms.testProbabilisticClassifier(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            ea.generateFFNDE(mlp, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:1],
                             prob_target=0.5, beta=0.5, population_size=40, max_generations=100)
            mlp_DE_1_missed.append(ms.testProbabilisticClassifier(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            PSO.generateFFNgBestPSO(mlp, training_set[:validation_index], training_set[validation_index:],
                                    [len(chunks[0][0]) - 1] + hidden_layer_nodes[:1],
                                    omega=0.8, c1=0.6, c2=0.6, population_size=50, max_time_steps=80)
            mlp_PSO_1_missed.append(ms.testProbabilisticClassifier(mlp, testing_set))
            # 2-layer
            mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            mlp.backpropogation(training_set, hidden_layer_nodes[:2], eta, alpha_momentum, iterations)
            mlp_BP_2_missed.append(ms.testProbabilisticClassifier(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            ea.generateFFNGA(mlp, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:2],
                             prob_cross=0.6, prob_mutation=0.01, mutation_variance=0.2, population_size=60,
                             max_generations=300)
            mlp_GA_2_missed.append(ms.testProbabilisticClassifier(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            ea.generateFFNDE(mlp, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:2],
                             prob_target=0.5, beta=0.5, population_size=40, max_generations=100)
            mlp_DE_2_missed.append(ms.testProbabilisticClassifier(mlp, testing_set))
            mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
            PSO.generateFFNgBestPSO(mlp, training_set[:validation_index], training_set[validation_index:],
                                    [len(chunks[0][0]) - 1] + hidden_layer_nodes[:2],
                                    omega=0.8, c1=0.6, c2=0.6, population_size=50, max_time_steps=80)
            mlp_PSO_2_missed.append(ms.testProbabilisticClassifier(mlp, testing_set))

    if use_regression:
        # 0-layer
        ms.compareRegressors(mlp_BP_0_missed, mlp_GA_0_missed, "BP 0-layer", "GA 0-layer")
        ms.compareRegressors(mlp_BP_0_missed, mlp_DE_0_missed, "BP 0-layer", "DE 0-layer")
        ms.compareRegressors(mlp_BP_0_missed, mlp_PSO_0_missed, "BP 0-layer", "PSO 0-layer")
        ms.compareRegressors(mlp_GA_0_missed, mlp_DE_0_missed, "GA 0-layer", "DE 0-layer")
        ms.compareRegressors(mlp_GA_0_missed, mlp_PSO_0_missed, "GA 0-layer", "PSO 0-layer")
        ms.compareRegressors(mlp_DE_0_missed, mlp_PSO_0_missed, "DE 0-layer", "PSO 0-layer")

        # 1-layer
        ms.compareRegressors(mlp_BP_1_missed, mlp_GA_1_missed, "BP 1-layer", "GA 1-layer")
        ms.compareRegressors(mlp_BP_1_missed, mlp_DE_1_missed, "BP 1-layer", "DE 1-layer")
        ms.compareRegressors(mlp_BP_1_missed, mlp_PSO_1_missed, "BP 1-layer", "PSO 1-layer")
        ms.compareRegressors(mlp_GA_1_missed, mlp_DE_1_missed, "GA 1-layer", "DE 1-layer")
        ms.compareRegressors(mlp_GA_1_missed, mlp_PSO_1_missed, "GA 1-layer", "PSO 1-layer")
        ms.compareRegressors(mlp_DE_1_missed, mlp_PSO_1_missed, "DE 1-layer", "PSO 1-layer")

        # 2-layer
        ms.compareRegressors(mlp_BP_2_missed, mlp_GA_2_missed, "BP 2-layer", "GA 2-layer")
        ms.compareRegressors(mlp_BP_2_missed, mlp_DE_2_missed, "BP 2-layer", "DE 2-layer")
        ms.compareRegressors(mlp_BP_2_missed, mlp_PSO_2_missed, "BP 2-layer", "PSO 2-layer")
        ms.compareRegressors(mlp_GA_2_missed, mlp_DE_2_missed, "GA 2-layer", "DE 2-layer")
        ms.compareRegressors(mlp_GA_2_missed, mlp_PSO_2_missed, "GA 2-layer", "PSO 2-layer")
        ms.compareRegressors(mlp_DE_2_missed, mlp_PSO_2_missed, "DE 2-layer", "PSO 2-layer")
    else:
        # 0-layer
        ms.compareProbabilisticClassifiers(mlp_BP_0_missed, mlp_GA_0_missed, "BP 0-layer", "GA 0-layer")
        ms.compareProbabilisticClassifiers(mlp_BP_0_missed, mlp_DE_0_missed, "BP 0-layer", "DE 0-layer")
        ms.compareProbabilisticClassifiers(mlp_BP_0_missed, mlp_PSO_0_missed, "BP 0-layer", "PSO 0-layer")
        ms.compareProbabilisticClassifiers(mlp_GA_0_missed, mlp_DE_0_missed, "GA 0-layer", "DE 0-layer")
        ms.compareProbabilisticClassifiers(mlp_GA_0_missed, mlp_PSO_0_missed, "GA 0-layer", "PSO 0-layer")
        ms.compareProbabilisticClassifiers(mlp_DE_0_missed, mlp_PSO_0_missed, "DE 0-layer", "PSO 0-layer")

        # 1-layer
        ms.compareProbabilisticClassifiers(mlp_BP_1_missed, mlp_GA_1_missed, "BP 1-layer", "GA 1-layer")
        ms.compareProbabilisticClassifiers(mlp_BP_1_missed, mlp_DE_1_missed, "BP 1-layer", "DE 1-layer")
        ms.compareProbabilisticClassifiers(mlp_BP_1_missed, mlp_PSO_1_missed, "BP 1-layer", "PSO 1-layer")
        ms.compareProbabilisticClassifiers(mlp_GA_1_missed, mlp_DE_1_missed, "GA 1-layer", "DE 1-layer")
        ms.compareProbabilisticClassifiers(mlp_GA_1_missed, mlp_PSO_1_missed, "GA 1-layer", "PSO 1-layer")
        ms.compareProbabilisticClassifiers(mlp_DE_1_missed, mlp_PSO_1_missed, "DE 1-layer", "PSO 1-layer")

        # 2-layer
        ms.compareProbabilisticClassifiers(mlp_BP_2_missed, mlp_GA_2_missed, "BP 2-layer", "GA 2-layer")
        ms.compareProbabilisticClassifiers(mlp_BP_2_missed, mlp_DE_2_missed, "BP 2-layer", "DE 2-layer")
        ms.compareProbabilisticClassifiers(mlp_BP_2_missed, mlp_PSO_2_missed, "BP 2-layer", "PSO 2-layer")
        ms.compareProbabilisticClassifiers(mlp_GA_2_missed, mlp_DE_2_missed, "GA 2-layer", "DE 2-layer")
        ms.compareProbabilisticClassifiers(mlp_GA_2_missed, mlp_PSO_2_missed, "GA 2-layer", "PSO 2-layer")
        ms.compareProbabilisticClassifiers(mlp_DE_2_missed, mlp_PSO_2_missed, "DE 2-layer", "PSO 2-layer")

if(len(sys.argv) > 3):
    chunks, class_list = openFiles(sys.argv[1])
    uses_regression = False
    tun = False
    if sys.argv[2] == 'r':
        print("Using regression")
        uses_regression = True
    else:
        print("Using classification")
    if sys.argv[3] == "tune":
        print("Tuning")
        tun = True

    print("Using k=3")
    hidden_layer_nodes = []
    for i in range(3):
        hidden_layer_nodes.append(6*(len(chunks[0][0])-1))
    trainAndTest(chunks, class_list, 3, uses_regression, 1, hidden_layer_nodes, 0.05, 0, 100, tun)
else:
    print("Usage:\t<dataFile.data> <r> <tune/notune>(for regression, use any other character for classification)")