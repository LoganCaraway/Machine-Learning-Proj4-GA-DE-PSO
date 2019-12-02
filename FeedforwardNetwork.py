import copy
import time
import MathAndStats as ms
import random
import Neuron as unit


class FeedforwardNetwork:

    # output_type is either "classification", "regression", or "autoencoder". Defaults to classification
    # logistic_nodes and logistic_output are boolean variables determining of the nodes are linear or logistic
    def __init__(self, out_k, clsses, output_type, logistic_nodes, logistic_output):
        self.hidden_layers = []
        self.output_layer = []
        self.output_type = output_type
        self.class_list = clsses
        self.logistic_nodes = logistic_nodes
        self.logistic_output = logistic_output
        self.ffn = None
        self.regularize = False
        if output_type == "autoencoder":
            self.name = "autoencoder"
        else:
            self.name = "mlp"
        # in the case of regression, overwrite these inputs
        if output_type == "regression":
            self.out_k = 1
            logistic_output = False
        else:
            self.out_k = out_k

    # given a list of nodes per layer in the network (counting input/output layers), set up the nodes
    def makeNodes(self, nodes_by_layer):
        pass
        self.hidden_layers = []
        for layer in range(1, len(nodes_by_layer)):
            if layer == 1:
                inputs = nodes_by_layer[0]
            else:
                inputs = nodes_by_layer[layer-1]
            # append a list for the nodes of this layer
            self.hidden_layers.append([])
            for node in range(nodes_by_layer[layer]):
                self.hidden_layers[layer-1].append(unit.Neuron(inputs, self.logistic_nodes))
        if len(nodes_by_layer) != 0:
            inputs = nodes_by_layer[-1]
        else:
            inputs = nodes_by_layer[0]
        # output layer
        self.output_layer = []
        for output_node in range(self.out_k):
            self.output_layer.append(unit.Neuron(inputs, self.logistic_output))
            if not ((self.output_type == "regression") or (self.output_type == "autoencoder")):
                self.output_layer[output_node].setClass(self.class_list[output_node])

    # given a list for all weights in the network, assign the weights to the nodes
    def setWeights(self, nodes_by_layer, weights):
        if (self.output_layer == []) or (self.hidden_layers == []):
            self.makeNodes(nodes_by_layer)
        weight_num = 0
        for layer in range(len(self.hidden_layers)):
            for node in range(len(self.hidden_layers[layer])):
                for weight_index in range(len(self.hidden_layers[layer][node].weights)):
                    self.hidden_layers[layer][node].weights[weight_index] = weights[weight_num]
                    weight_num += 1
        for output_node in range(len(self.output_layer)):
            for weight_index in range(len(self.output_layer[output_node].weights)):
                self.output_layer[output_node].weights[weight_index] = weights[weight_num]
                weight_num += 1

    def getHiddenLayerOutput(self, new_obs, layer_num):
        data = []
        # bias node
        data.append(1.0)
        for hidden_node_num in range(len(self.hidden_layers[layer_num])):
            data.append(self.hidden_layers[layer_num][hidden_node_num].getOutput(new_obs))
        return data

    def tuneLayerwise(self, input_data, eta, alpha_momentum, iterations):
        print("Tuning by Layer...")
        for layer_num in range(len(self.hidden_layers)):
            print("Layer ",layer_num+1,"of",len(self.hidden_layers))
            if layer_num == 0:
                inputs = len(input_data[0])-1
            else:
                inputs = len(self.hidden_layers[layer_num-1])
            # create outputs for this layer
            for output_node in range(inputs):
                self.output_layer.append(unit.Neuron(len(self.hidden_layers[layer_num]), self.logistic_output))
            # tune layer
            random.shuffle(input_data)
            prev_delta_weights = []
            prev_loss = []
            worse_epochs = 0
            for epoch in range(iterations):
                loss = [0.0] * len(self.output_layer)
                delta_weights = []
                for example_num in range(len(input_data)):
                    delta_weights.append([])
                    # get hidden outputs
                    hidden_outputs = []
                    for layer in range(layer_num+1):
                        if layer == 0:
                            hidden_outputs.append(self.getHiddenLayerOutput(input_data[example_num][:-1], 0))
                        else:
                            # else, take outputs from previous layer
                            hidden_outputs.append(self.getHiddenLayerOutput(hidden_outputs[-1], layer))
                    # get outputs
                    outputs = []
                    #if hidden_outputs == []:
                    #    hidden_outputs.append(input_data[example_num][:-1])
                    for output_node in range(len(self.output_layer)):
                        outputs.append(self.output_layer[output_node].getOutput(hidden_outputs[-1]))
                    # -------------#

                    # prev_error[node]
                    prev_error = []
                    # prev_weights[node][weight]
                    prev_weights = []
                    # append a list for output layer
                    delta_weights[example_num].append([])
                    for output_node in range(len(self.output_layer)):
                        # shallow copy the weights (alias)
                        weights = self.output_layer[output_node].weights
                        prev_weights.append(copy.deepcopy(weights))
                        # append a list to hold the weights for this node
                        delta_weights[example_num][0].append([])
                        if not self.regularize:
                            if layer_num != 0:
                                error = hidden_outputs[-2][output_node] - outputs[output_node]
                            else:
                                # error = feature val - predicted feature val
                                error = input_data[example_num][output_node] - outputs[output_node]
                        else:
                            sum_of_weights = 0
                            for weight_num in range(len(weights)):
                                sum_of_weights += abs(weights[weight_num])
                            if layer_num != 0:
                                error = hidden_outputs[-2][output_node] - outputs[output_node] + (self.lmbda * sum_of_weights)
                            else:
                                # error = feature val - predicted feature val + lambda*(sum of weights)
                                error = input_data[example_num][output_node] - outputs[output_node] + (self.lmbda * sum_of_weights)
                        prev_error.append(error)
                        if layer_num == 0:
                            loss[output_node] += ms.getDecimalSMAPE(input_data[example_num][output_node], outputs[output_node])
                        else:
                            loss[output_node] += ms.getDecimalSMAPE(hidden_outputs[-2][output_node], outputs[output_node])
                        for weight_num in range(len(weights)):
                            delta_weights[example_num][0][output_node].append(eta * error * hidden_outputs[-1][weight_num])
                            if (alpha_momentum > 0) and (example_num > 0):
                                weights[weight_num] = weights[weight_num] + delta_weights[example_num][0][output_node][weight_num] + (alpha_momentum * prev_delta_weights[0][output_node][weight_num])
                            else:
                                weights[weight_num] = weights[weight_num] + delta_weights[example_num][0][output_node][weight_num]
                    # propogate errors and update weights
                    #current_error = []
                    #current_weights = []
                    # append a list for hidden layer
                    delta_weights[example_num].append([])
                    for node in range(len(self.hidden_layers[layer_num])):
                        # add list for the weights of this node
                        delta_weights[example_num][1].append([])
                        sum_downsteam_error = 0.0
                        for downstream_node in range(len(prev_error)):
                            sum_downsteam_error += prev_error[downstream_node] * prev_weights[downstream_node][node + 1]
                        activation = hidden_outputs[-1][node + 1]
                        if self.logistic_nodes:
                            current_error = sum_downsteam_error * activation * (1 - activation)
                        else:
                            current_error = sum_downsteam_error

                        # shallow copy the weights (alias)
                        weights = self.hidden_layers[layer_num][node].weights
                        #current_weights.append(copy.deepcopy(weights))
                        for weight_num in range(len(weights)):
                            if layer_num != 0:
                                input = hidden_outputs[-2][weight_num]
                            else:
                                if weight_num == 0:
                                    input = 1
                                else:
                                    input = input_data[example_num][weight_num - 1]
                            #if layer_num != 0:
                            #    delta_weights[example_num][1][node].append(eta * current_error * (hidden_outputs[-2][weight_num]))
                            #else:
                            #    delta_weights[example_num][1][node].append(eta * current_error * (input_data[example_num][weight_num]))
                            delta_weights[example_num][1][node].append(eta * current_error * input)
                            if (alpha_momentum > 0) and (example_num > 0):
                                weights[weight_num] = weights[weight_num] + delta_weights[example_num][1][node][weight_num] + alpha_momentum * prev_delta_weights[1][node][weight_num]
                            else:
                                weights[weight_num] = weights[weight_num] + delta_weights[example_num][1][node][weight_num]
                        #prev_error = current_error
                        #prev_weights = current_weights
                    if alpha_momentum > 0:
                        prev_delta_weights = delta_weights[-1]
                print(self.name,":",epoch+1,"of",iterations, end=' ')
                average = 0.0
                for output_num in range(len(self.output_layer)):
                    average += loss[output_num]
                # take average and convert decimal to %
                average /= len(input_data)
                average *= (100 / len(self.output_layer))
                if self.regularize:
                    print("Regularized ", end='')
                print("Average Symmetric Mean Absolute Percentage Error:", average)

                if epoch > 10:
                    prev_average = 0.0
                    for output_num in range(len(self.output_layer)):
                        prev_average += prev_loss[output_num]
                    # take average and convert decimal to %
                    prev_average /= len(input_data)
                    prev_average *= (100 / len(self.output_layer))
                    if average > prev_average:
                        worse_epochs += 1
                        if (worse_epochs > 4) or (average > (9*prev_average)):
                            print("Converged")
                            break
                    else:
                        worse_epochs -= 2
                        if worse_epochs < 0:
                            worse_epochs = 0
                prev_loss = loss
            # clear outputlayer
            self.output_layer = []


    def backpropogation(self, input_data, hidden_layer_nodes, eta, alpha_momentum, iterations):
        print("Training",self.name,"using: eta =",eta,", alpha =",alpha_momentum,", nodes by layer =",hidden_layer_nodes)
        for layer in range(len(hidden_layer_nodes)):
            if layer == 0:
                # if first hidden layer, number of inputs is number of features
                inputs = len(input_data[0])-1
            else:
                # else number of inputs is number of nodes from previous layer (bias added automatically)
                inputs = len(self.hidden_layers[-1])
            # append a list for the nodes of this layer
            self.hidden_layers.append([])
            for node in range(hidden_layer_nodes[layer]):
                self.hidden_layers[layer].append(unit.Neuron(inputs, self.logistic_nodes))
        if (self.output_type == "autoencoder") and (len(hidden_layer_nodes) > 1):
            self.tuneLayerwise(input_data, eta, alpha_momentum, iterations)
            print("Fine tuning weights using Backpropogation")
        #-create output nodes-#
        # if there is at least one hidden layer, set number of inputs to the number of nodes in the final layer
        if len(hidden_layer_nodes) != 0:
            inputs = len(self.hidden_layers[-1])
        else:
            # set inputs equal to number of features of input data
            inputs = len(input_data[0]) - 1
        for output_node in range(self.out_k):
            self.output_layer.append(unit.Neuron(inputs, self.logistic_output))
            if not ((self.output_type == "regression") or (self.output_type == "autoencoder")):
                self.output_layer[output_node].setClass(self.class_list[output_node])
        #-----------------#
        # Backpropogation #
        #-----------------#
        random.shuffle(input_data)
        prev_delta_weights = []
        prev_loss = []
        worse_epochs = 0
        for epoch in range(iterations):
            loss = [0.0] * self.out_k
            # delta_weights[example_num][layer][node][weight]
            delta_weights = []
            for example_num in range(len(input_data)):
                delta_weights.append([])
                #-------------#
                #-Get Outputs-#
                # -------------#
                # get the output for each hidden node of each hidden layer
                # hidden_outputs[layer][node]
                hidden_outputs = []
                if len(self.hidden_layers) > 0:
                    for layer in range(len(self.hidden_layers)):
                        # if first layer, take data inputs
                        if layer == 0:
                            hidden_outputs.append(self.getHiddenLayerOutput(input_data[example_num][:-1], 0))
                        else:
                            # else, take outputs from previous layer
                            hidden_outputs.append(self.getHiddenLayerOutput(hidden_outputs[-1], layer))
                # there are no hidden layers
                else:
                    # add bias node
                    hidden_outputs.append([1] + input_data[example_num][:-1])
                # get outputs by layer
                # outputs [layer][node]
                outputs = []
                for output_node in range(self.out_k):
                    outputs.append(self.output_layer[output_node].getOutput(hidden_outputs[-1]))
                # -------------#

                #-get output node errors and update output weights-#
                #--------------------------------------------------#
                # prev_error[node]
                prev_error = []
                # prev_weights[node][weight]
                prev_weights = []
                # append a list for output layer
                delta_weights[example_num].append([])
                for output_node in range(self.out_k):
                    # shallow copy the weights (alias)
                    weights = self.output_layer[output_node].weights
                    prev_weights.append(copy.deepcopy(weights))
                    # append a list to hold the weights for this node
                    delta_weights[example_num][0].append([])
                    error = 0
                    if self.output_type == "regression":
                        error = input_data[example_num][-1] - outputs[output_node]
                        loss[output_node] += error * error
                    elif self.output_type == "classification":
                        if input_data[example_num][-1] == self.output_layer[output_node].clss:
                            correct = 1
                        else:
                            correct = 0
                        error = correct - outputs[output_node]
                        error *= outputs[output_node] * (1-outputs[output_node])
                        loss[output_node] += error * error
                    elif self.output_type == "autoencoder":
                        if not self.regularize:
                            # error = feature val - predicted feature val
                            error = input_data[example_num][output_node] - outputs[output_node]
                        else:
                            sum_of_weights = 0
                            for weight_num in range(len(weights)):
                                sum_of_weights += abs(weights[weight_num])
                            # error = feature val - predicted feature val + lambda*(sum of weights)
                            error = input_data[example_num][output_node] - outputs[output_node] + (self.lmbda * sum_of_weights)
                        loss[output_node] += ms.getDecimalSMAPE(input_data[example_num][output_node], outputs[output_node])
                    prev_error.append(error)
                    for weight_num in range(len(weights)):
                        delta_weights[example_num][0][output_node].append(eta * error * hidden_outputs[-1][weight_num])
                        if (alpha_momentum > 0) and (example_num > 0):
                            weights[weight_num] = weights[weight_num] + delta_weights[example_num][0][output_node][weight_num] + (alpha_momentum * prev_delta_weights[0][output_node][weight_num])
                        else:
                            weights[weight_num] = weights[weight_num] + delta_weights[example_num][0][output_node][weight_num]
                # propogate errors and update weights
                # start at final layer and move backwards
                prev_layer_error = []
                for layer in range(len(self.hidden_layers)):
                    delta_weights[example_num].append([])
                for layer in range(len(self.hidden_layers)-1, -1, -1):
                    # check if the downstream layer is the outputs
                    if layer == (len(self.hidden_layers)-1):
                        prev_layer_error = prev_error
                    current_error = []
                    current_weights = []
                    # iterate through the nodes in a given layer
                    for node in range(len(self.hidden_layers[layer])):
                        # add list for the weights of this node
                        delta_weights[example_num][layer+1].append([])
                        sum_downsteam_error = 0.0
                        for downstream_node in range(len(prev_layer_error)):
                            sum_downsteam_error += prev_layer_error[downstream_node] * prev_weights[downstream_node][node+1]
                        activation = hidden_outputs[layer][node+1]
                        if self.logistic_nodes:
                            current_error.append(sum_downsteam_error * activation * (1-activation))
                        else:
                            current_error.append(sum_downsteam_error)

                        # shallow copy the weights (alias)
                        weights = self.hidden_layers[layer][node].weights
                        current_weights.append(copy.deepcopy(weights))
                        for weight_num in range(len(weights)):
                            if layer != 0:
                                input = hidden_outputs[layer - 1][weight_num]
                            else:
                                if weight_num == 0:
                                    input = 1
                                else:
                                    input = input_data[example_num][weight_num - 1]
                            delta_weights[example_num][layer + 1][node].append(eta * current_error[node] * input)
                            #if layer != 0:
                            #    delta_weights[example_num][layer+1][node].append(eta * current_error[node] * (hidden_outputs[layer-1][weight_num]))
                            #else:
                            #    delta_weights[example_num][layer+1][node].append(eta * current_error[node] * (input_data[example_num][weight_num]))
                            if (alpha_momentum > 0) and (example_num > 0):
                                weights[weight_num] = weights[weight_num] + delta_weights[example_num][layer+1][node][weight_num] + alpha_momentum * prev_delta_weights[layer+1][node][weight_num]
                            else:
                                weights[weight_num] = weights[weight_num] + delta_weights[example_num][layer+1][node][weight_num]
                    prev_layer_error = current_error
                    prev_weights = current_weights
                if alpha_momentum > 0:
                    prev_delta_weights = delta_weights[-1]
            if self.output_type == "regression":
                print(self.name,":",epoch+1,"of",iterations, end=' ')
                print("MSE:", (loss[0] / len(input_data)))
            elif self.output_type == "autoencoder":
                print(self.name, ":", epoch + 1, "of", iterations, end=' ')
                average = 0.0
                for output_num in range(len(self.output_layer)):
                    average += loss[output_num]
                # take average and convert decimal to %
                average /= len(input_data)
                average *= (100 / len(self.output_layer))
                if self.regularize:
                    print("Regularized ", end='')
                print("Average Symmetric Mean Absolute Percentage Error:", average)
            else:
                print(self.name,":",epoch+1,"of",iterations, end=' ')
                average = 0.0
                for output_num in range(len(self.output_layer)):
                    average += loss[output_num] / len(input_data)
                average /= len(self.output_layer)
                print("Average MSE:", average)

            if epoch > 10:
                prev_error = 0.0
                current_error = 0.0
                for output_num in range(len(self.output_layer)):
                    current_error += loss[output_num]
                    prev_error += prev_loss[output_num]
                if current_error > prev_error:
                    worse_epochs += 1
                    if (worse_epochs > 4) or (current_error > (9*prev_error)):
                        print("Converged")
                        break
                else:
                    worse_epochs -= 2
                    if worse_epochs < 0:
                        worse_epochs = 0
            prev_loss = loss

    def tune(self, input_data, validation_data, num_layers, hidden_layer_nodes, tuning_iterations, iterations):
        eta = 0.05
        alpha = 0
        self.lmbda = 0.01
        if self.regularize:
            if hidden_layer_nodes == []:
                for layer in range(abs(num_layers)):
                    # while tuning lambda, set layers to be overcomplete
                    hidden_layer_nodes.append((len(input_data[0])-1))
            while self.lmbda <= 0.1:
                print("Testing lambda: ",self.lmbda)
                self.train(input_data, hidden_layer_nodes, eta, alpha, int(iterations/4))
                error = ms.testAutoencoder(self, validation_data)
                print("SMAPE for lambda =", self.lmbda, ":", error)
                if (error > self.allowedError):
                    self.lmbda -= 0.01
                    if self.lmbda < 0.01:
                        self.lmbda = 0.01
                    self.hidden_layers = []
                    self.output_layer = []
                    break
                self.lmbda += 0.01
                if self.lmbda > 0.1:
                    self.lmbda = 0.1
                self.hidden_layers = []
                self.output_layer = []
            print("Selected", self.name, "lambda =", self.lmbda)
            hidden_layer_nodes = []
        if hidden_layer_nodes == []:
            print("Tuning",self.name,"nodes per layer for",num_layers,"layers")
            for layer in range(num_layers):
                less_nodes = 1
                if self.output_type == "autoencoder":
                    if self.regularize:
                        # if regularized, set max nodes to 3x the number of features
                        more_nodes = int(3*(len(input_data[0])-1))
                    else:
                        # if not regularize, ensure there are less nodes than features
                        more_nodes = len(input_data[0]) - 2
                else:
                    more_nodes = 100
                nodes = random.randint(less_nodes+1, more_nodes-1)
                max_loops = 4
                for round in range(max_loops):
                    less_error = 0
                    more_error = 0
                    mid_error = 0
                    hidden_layer_nodes.append(nodes)
                    self.train(input_data, hidden_layer_nodes, eta, alpha, tuning_iterations)
                    del hidden_layer_nodes[-1]
                    if self.output_type == "regression":
                        # get absolute error for test
                        results = ms.testRegressor(self, validation_data)
                        for obs in range(len(results)):
                            mid_error += (results[obs] * results[obs])
                        mid_error /= len(results)
                    elif self.output_type == "classification":
                        mid_error = self.testClassification(validation_data)
                    else:
                        mid_error = ms.testAutoencoder(self, validation_data)
                    self.hidden_layers = []
                    self.output_layer = []
                    hidden_layer_nodes.append(more_nodes)
                    self.train(input_data, hidden_layer_nodes, eta, alpha, tuning_iterations)
                    del hidden_layer_nodes[-1]
                    if self.output_type == "regression":
                        # get absolute error for test
                        results = ms.testRegressor(self, validation_data)
                        for obs in range(len(results)):
                            more_error += (results[obs] * results[obs])
                        more_error /= len(results)
                    elif self.output_type == "classification":
                        more_error = self.testClassification(validation_data)
                    else:
                        more_error = ms.testAutoencoder(self, validation_data)
                    self.hidden_layers = []
                    self.output_layer = []
                    hidden_layer_nodes.append(less_nodes)
                    self.train(input_data, hidden_layer_nodes, eta, alpha, tuning_iterations)
                    del hidden_layer_nodes[-1]
                    if self.output_type == "regression":
                        # get error for test
                        results = ms.testRegressor(self, validation_data)
                        for obs in range(len(results)):
                            less_error += (results[obs] * results[obs])
                        less_error /= len(results)
                    elif self.output_type == "classification":
                        less_error = self.testClassification(validation_data)
                    else:
                        less_error = ms.testAutoencoder(self, validation_data)
                    self.hidden_layers = []
                    self.output_layer = []
                    if (mid_error <= less_error) and (mid_error <= more_error) and round > 0:
                        break
                    elif (more_error <= mid_error) and (more_error <= less_error):
                        old_nodes = nodes
                        try:
                            nodes = random.randint(nodes+1, more_nodes-1)
                            less_nodes = random.randint(old_nodes+1, nodes-1)
                            more_nodes = random.randint(nodes+1, more_nodes-1)
                        except:
                            nodes = old_nodes
                            break
                    elif (less_error <= mid_error) and (less_error <= more_error):
                        old_nodes = nodes
                        try:
                            nodes = random.randint(less_nodes+1, nodes-1)
                            less_nodes = random.randint(less_nodes+1, nodes-1)
                            more_nodes = random.randint(nodes+1, old_nodes-1)
                        except:
                            nodes = old_nodes
                            break
                    else:
                        if more_error <= less_error:
                            old_nodes = nodes
                            try:
                                nodes = random.randint(nodes + 1, more_nodes - 1)
                                less_nodes = random.randint(old_nodes + 1, nodes - 1)
                                more_nodes = random.randint(nodes + 1, more_nodes - 1)
                            except:
                                nodes = old_nodes
                                break
                        else:
                            old_nodes = nodes
                            try:
                                nodes = random.randint(less_nodes + 1, nodes - 1)
                                less_nodes = random.randint(less_nodes + 1, nodes - 1)
                                more_nodes = random.randint(nodes + 1, old_nodes - 1)
                            except:
                                nodes = old_nodes
                                break
                hidden_layer_nodes.append(nodes)
            print("Selected nodes by layer: ",hidden_layer_nodes)
        lowest_eta = -1
        lowest_error = -1
        print("Tuning",self.name,"eta")
        while eta <= 0.3:
            self.train(input_data, hidden_layer_nodes, eta, alpha, tuning_iterations)
            error = 0
            if self.output_type == "regression":
                # get error for test
                results = ms.testRegressor(self, validation_data)
                for obs in range(len(results)):
                    error += (results[obs] * results[obs])
                error /= len(results)
            elif self.output_type == "classification":
                error = self.testClassification(validation_data)
            elif self.output_type == "autoencoder":
                # get error for test
                error = ms.testAutoencoder(self, validation_data)
            print("Error for eta =", eta, ":", error, end=' ')
            if lowest_error == -1:
                print()
            else:
                print(" lowest error =", lowest_error)
            if (error < lowest_error) or (eta == 0.05):
                lowest_eta = eta
                lowest_error = error
            eta += 0.05
            self.hidden_layers = []
            self.output_layer = []
        print("Selected",self.name,"eta =", lowest_eta)

        print("Tuning",self.name,"alpha for momentum")
        lowest_alpha = 0
        lowest_error = -1
        while alpha < 0.3:
            self.train(input_data, hidden_layer_nodes, lowest_eta, alpha, tuning_iterations)
            error = 0
            if self.output_type == "regression":
                # get error for test
                results = ms.testRegressor(self, validation_data)
                for obs in range(len(results)):
                    error += (results[obs] * results[obs])
                error /= len(results)
            elif self.output_type == "classification":
                error = self.testClassification(validation_data)
            elif self.output_type == "autoencoder":
                # get error for test
                error = ms.testAutoencoder(self, validation_data)
            print("Error for alpha =", alpha, ":", error, end=' ')
            if lowest_error == -1:
                print()
            else:
                print(" lowest error =", lowest_error)
            if (error < lowest_error) or (alpha == 0):
                lowest_alpha = alpha
                lowest_error = error
            prev_error = error
            alpha += 0.1
            self.hidden_layers = []
            self.output_layer = []
        print("Selected",self.name,"alpha =", lowest_alpha)
        now = time.time()
        self.train(input_data, hidden_layer_nodes, lowest_eta, lowest_alpha, iterations)
        done = time.time()
        self.convergence_time = done - now

    def regularizeAutoencoder(self, lmbda):
        self.regularize = True
        self.lmbda = lmbda
        #self.allowedError = max_error

    def predict(self, new_obs):
        if self.output_type == "regression":
            return self.regress(new_obs)
        elif self.output_type == "autoencoder":
            return self.reproduce(new_obs)
        else:
            # I have classify return (class, probability) as a tuple for use in tuning, but
            # predict will simply return class
            return self.classify(new_obs)[0]

    def reproduce(self, new_obs):
        # get the output for each hidden node of each hidden layer
        hidden_outputs = []
        if len(self.hidden_layers) > 0:
            for layer in range(len(self.hidden_layers)):
                # if first layer, take data inputs
                if layer == 0:
                    hidden_outputs = self.getHiddenLayerOutput(new_obs, 0)
                else:
                    # else, take outputs from previous layer
                    hidden_outputs = self.getHiddenLayerOutput(hidden_outputs, layer)
        else:
            # there are no hidden layers
            hidden_outputs = new_obs
        outputs = []
        for output_node in range(self.out_k):
            outputs.append(self.output_layer[output_node].getOutput(hidden_outputs))
        if self.ffn is None:
            return outputs
        # if there is a feedforward network stacked ontop, return the result from that
        return self.ffn.predict(outputs)

    def regress(self, new_obs):
        # get the output for each hidden node of each hidden layer
        hidden_outputs = []
        if len(self.hidden_layers) > 0:
            for layer in range(len(self.hidden_layers)):
                # if first layer, take data inputs
                if layer == 0:
                    hidden_outputs = self.getHiddenLayerOutput(new_obs, 0)
                else:
                    # else, take outputs from previous layer
                    hidden_outputs = self.getHiddenLayerOutput(hidden_outputs, layer)
        else:
            # there are no hidden layers
            hidden_outputs = new_obs
        return self.output_layer[0].getOutput(hidden_outputs)

    def classify(self, new_obs):
        if self.output_type == "autoencoder":
            return self.ffn.classify(new_obs)
        hidden_outputs = []
        if len(self.hidden_layers) > 0:
            for layer in range(len(self.hidden_layers)):
                # if first layer, take data inputs
                if layer == 0:
                    hidden_outputs = self.getHiddenLayerOutput(new_obs, 0)
                else:
                    # else, take outputs from previous layer
                    hidden_outputs = self.getHiddenLayerOutput(hidden_outputs, layer)
        else:
            # there are no hidden layers
            hidden_outputs = new_obs
        #return self.output_layer[0].getOutput(hidden_outputs)
        classes = {}
        for output_num in range(len(self.output_layer)):
            classes[self.output_layer[output_num].clss] = self.output_layer[output_num].getOutput(hidden_outputs)
        decision = sorted(classes.items(), key=lambda elem: elem[1], reverse=True)
        return decision[0]

    # will return the MSE for classification
    def testClassification(self, testing_set):
        mse = 0
        for obs in range(len(testing_set)):
            correct_class = testing_set[obs][-1]
            predicted = self.classify(testing_set[obs][:-1])
            if predicted[0] == correct_class:
                error = (1 - predicted[1])
            else:
                error = predicted[1]
            mse += error * error
        return mse / len(testing_set)

    def addFFNetwork(self, ff, tune, training_set, hidden_layer_nodes, eta, alpha_momentum, iterations):
        if self.output_type == "autoencoder":
            # if the given network is already tuned, simply place on top of the autoencoder
            if not tune:
                self.ffn = ff
                return
            training_output = []
            for example in range(len(training_set)):
                predicted = self.predict(training_set[example][:-1])
                predicted.append(training_set[example][-1])
                training_output.append(predicted)
            #validation_output = []
            #for example in range(len(validation_set)):
            #    predicted = self.predict(validation_set[example][:-1])
            #    predicted.append(training_set[example][-1])
            #    validation_output.append(predicted)
            self.ffn = ff
            # setting num_layers to -1 signifies that the autoencoder should train a 0, 1, and 2 layer FFN
            # and pick the best
            #if num_layers == -1:
            #    pass
            #else:
            print("Tuning stacked FFN on top of",len(self.hidden_layers),"layer autoencoder")
            #self.ffn.tune(training_output, validation_output, num_layers, [], tuning_iterations, iterations)
            self.ffn.backpropogation(training_output, hidden_layer_nodes, eta, alpha_momentum, iterations)