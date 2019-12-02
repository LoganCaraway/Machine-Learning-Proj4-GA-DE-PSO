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
        print("Fold: ",testing,"of 10 fold cross validation")
        training_set = []

        testing_set = chunked_data[testing]
        # make example set
        for train in range(10):
            if train != testing:
                for x in range(len(chunked_data[train])):
                    training_set.append(chunked_data[train][x])

        validation_index = int((float(len(training_set)) * 8 / 10)) - 1
        if use_regression:
            mlp_GA_0 = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            ea.generateFFNGA(mlp_GA_0, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:0],
                             0.5, 0.5, 0.1, 15, 130)
            mlp_GA_1 = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            ea.generateFFNGA(mlp_GA_1, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:1],
                             0.5, 0.5, 0.1, 15, 130)
            mlp_GA_2 = ffn.FeedforwardNetwork(1, clss_list, "regression", True, False)
            ea.generateFFNGA(mlp_GA_2, training_set[:validation_index], training_set[validation_index:],
                             [len(chunks[0][0]) - 1] + hidden_layer_nodes[:2],
                             0.5, 0.5, 0.1, 15, 130)
            mlp_GA_0_missed.append(ms.testRegressor(mlp_GA_0, testing_set))
            mlp_GA_1_missed.append(ms.testRegressor(mlp_GA_1, testing_set))
            mlp_GA_2_missed.append(ms.testRegressor(mlp_GA_2, testing_set))
        # classification
        else:
            if tune:
                # train multi layer perceptrons
                mlp_0 = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
                mlp_0.backpropogation(training_set, [], eta, alpha_momentum, iterations)
                mlp_1 = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
                mlp_1.backpropogation(training_set, hidden_layer_nodes[:1], eta, alpha_momentum, iterations)
                mlp_2 = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
                mlp_2.backpropogation(training_set, hidden_layer_nodes[:2], eta, alpha_momentum, iterations)
                # test multi layer perceptrons
                mlp_0_missed.append(ms.testProbabilisticClassifier(mlp_0, testing_set))
                mlp_1_missed.append(ms.testProbabilisticClassifier(mlp_1, testing_set))
                mlp_2_missed.append(ms.testProbabilisticClassifier(mlp_2, testing_set))
            else:
                mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
                mlp.backpropogation(training_set, hidden_layer_nodes[:num_layers], eta, alpha_momentum, iterations)
                best_mlp_missed.append(ms.testProbabilisticClassifier(mlp, testing_set))
            # train autoencoders
            ae_1 = ffn.FeedforwardNetwork(len(chunked_data[0][0])-1, clss_list, "autoencoder", True, False)
            ae_1.regularizeAutoencoder(lmbda)
            ae_1.backpropogation(training_set, hidden_layer_nodes[:1], eta, alpha_momentum, iterations)
            # Autoencoder 1 stacked MLP
            if not tune:
                ae_1_mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
                ae_1.addFFNetwork(ae_1_mlp, True, training_set, hidden_layer_nodes[:num_layers], eta, alpha_momentum, iterations)

            ae_2 = ffn.FeedforwardNetwork(len(chunked_data[0][0])-1, clss_list, "autoencoder", True, False)
            ae_2.regularizeAutoencoder(lmbda)
            ae_2.backpropogation(training_set, hidden_layer_nodes[:2], eta, alpha_momentum, iterations)
            # Autoencoder 2 stacked MLP
            if not tune:
                ae_2_mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
                ae_2.addFFNetwork(ae_2_mlp, True, training_set, hidden_layer_nodes[:num_layers], eta, alpha_momentum, iterations)

            ae_3 = ffn.FeedforwardNetwork(len(chunked_data[0][0])-1, clss_list, "autoencoder", True, False)
            ae_3.regularizeAutoencoder(lmbda)
            #ae_3.tune(training_set[:validation_index], training_set[validation_index:], 3, [], 15, 150)
            ae_3.backpropogation(training_set, hidden_layer_nodes, eta, alpha_momentum, iterations)
            # Autoencoder 3 stacked MLP
            if not tune:
                ae_3_mlp = ffn.FeedforwardNetwork(len(clss_list), clss_list, "classification", True, True)
                ae_3.addFFNetwork(ae_3_mlp, True, training_set, hidden_layer_nodes[:num_layers], eta, alpha_momentum, iterations)

            # test autoencoders
            if tune:
                ae_1_smape.append(ms.testAutoencoder(ae_1, testing_set))
                ae_2_smape.append(ms.testAutoencoder(ae_2, testing_set))
                ae_3_smape.append(ms.testAutoencoder(ae_3, testing_set))
            else:
                ae_1_mlp_missed.append(ms.testProbabilisticClassifier(ae_1, testing_set))
                ae_2_mlp_missed.append(ms.testProbabilisticClassifier(ae_2, testing_set))
                ae_3_mlp_missed.append(ms.testProbabilisticClassifier(ae_3, testing_set))
    if use_regression:
        if tune:
            ms.compareRegressors(mlp_GA_0_missed, mlp_GA_1_missed, "MLP 0-layer", "MLP 1-layer")
            #ms.compareRegressors(mlp_0_missed, mlp_2_missed, "MLP 0-layer", "MLP 2-layer")
            #ms.compareRegressors(mlp_1_missed, mlp_2_missed, "MLP 1-layer", "MLP 2-layer")

        else:
            ms.compareRegressors(best_mlp_missed, ae_1_mlp_missed, "MLP", "MLP stacked on 1-layer Autoencoder")
            ms.compareRegressors(best_mlp_missed, ae_2_mlp_missed, "MLP", "MLP stacked on 2-layer Autoencoder")
            ms.compareRegressors(best_mlp_missed, ae_3_mlp_missed, "MLP", "MLP stacked on 3-layer Autoencoder")

            ms.compareRegressors(ae_1_mlp_missed, ae_2_mlp_missed, "MLP stacked on 1-layer Autoencoder", "MLP stacked on 2-layer Autoencoder")
            ms.compareRegressors(ae_1_mlp_missed, ae_3_mlp_missed, "MLP stacked on 1-layer Autoencoder", "MLP stacked on 3-layer Autoencoder")

            ms.compareRegressors(ae_2_mlp_missed, ae_3_mlp_missed, "MLP stacked on 2-layer Autoencoder","MLP stacked on 3-layer Autoencoder")
    else:
        if tune:
            ms.compareProbabilisticClassifiers(mlp_0_missed, mlp_1_missed, "MLP 0-layer", "MLP 1-layer")
            ms.compareProbabilisticClassifiers(mlp_0_missed, mlp_2_missed, "MLP 0-layer", "MLP 2-layer")
            ms.compareProbabilisticClassifiers(mlp_1_missed, mlp_2_missed, "MLP 1-layer", "MLP 2-layer")
            ae_1_smape_avg = 0
            ae_2_smape_avg = 0
            ae_3_smape_avg = 0
            for i in range(10):
                ae_1_smape_avg += ae_1_smape[i]
                ae_2_smape_avg += ae_2_smape[i]
                ae_3_smape_avg += ae_3_smape[i]
            print("Autoencoder symmetric mean absolute percentage error:", ae_1_smape_avg / 10, ae_2_smape_avg / 10, ae_3_smape_avg / 10)
        else:
            ms.compareProbabilisticClassifiers(best_mlp_missed, ae_1_mlp_missed, "MLP", "MLP stacked on 1-layer Autoencoder")
            ms.compareProbabilisticClassifiers(best_mlp_missed, ae_2_mlp_missed, "MLP", "MLP stacked on 2-layer Autoencoder")
            ms.compareProbabilisticClassifiers(best_mlp_missed, ae_3_mlp_missed, "MLP", "MLP stacked on 3-layer Autoencoder")

            ms.compareProbabilisticClassifiers(ae_1_mlp_missed, ae_2_mlp_missed, "MLP stacked on 1-layer Autoencoder", "MLP stacked on 2-layer Autoencoder")
            ms.compareProbabilisticClassifiers(ae_1_mlp_missed, ae_3_mlp_missed, "MLP stacked on 1-layer Autoencoder", "MLP stacked on 3-layer Autoencoder")

            ms.compareProbabilisticClassifiers(ae_2_mlp_missed, ae_3_mlp_missed, "MLP stacked on 2-layer Autoencoder", "MLP stacked on 3-layer Autoencoder")


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
        hidden_layer_nodes.append(5*(len(chunks[0][0])-1))
    trainAndTest(chunks, class_list, 3, uses_regression, 1, hidden_layer_nodes, 0.3, 0.2, 200, tun)
else:
    print("Usage:\t<dataFile.data> <r> <tune/notune>(for regression, use any other character for classification)")