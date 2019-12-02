import math
import numpy

# returns the squared distance between obs1 and obs 2
def squaredDistance(obs1, obs2, num_features):
    dist = 0.0
    for x in range(num_features):
        dist += pow((float(obs1[x]) - float(obs2[x])), 2)
    return dist

def weightedSum(vector, weights, lengt):
    sum = 0.0
    for element in range(lengt):
        sum += (vector[element] * weights[element])
    return sum

# returns the average value of an array
def getMean(arr, num_features):
    avg = 0.0
    for i in range(num_features):
        avg += float(arr[i])
    return avg/num_features

def getMeanValofTuple(arr, index):
    avg = 0.0
    for i in range(len(arr)):
        avg += float(arr[i][index])
    return avg/len(arr)

# returns the variance
def getVariance(mean, examples, num_features):
    if len(examples) < 2:
        print("Fewer than two examples: unable to calculate variance")
    # calculate the sum of squared distances between each point and the mean
    variance = 0.0
    for example_num in range(len(examples)):
        variance += squaredDistance(mean, examples[example_num], num_features)
    # divide by the number of observations - 1 (for unbiased variance)
    variance /= (len(examples) - 1)
    if variance == 0:
        variance = 0.01
    return variance

def logistic(x):
    if x < 0:
        return 1 - 1.0/(1 + math.exp(x))
    else:
        return 1.0/(1+math.exp(-x))

def normalize(data):
    # the final row is the classes, so we skip it
    for feature_num in range(len(data[0]) - 2, -1, -1):
        min = data[0][feature_num]
        max = data[0][feature_num]
        for example in range(len(data)):
            if data[example][feature_num] < min:
                min = data[example][feature_num]
            elif data[example][feature_num] > max:
                max = data[example][feature_num]
        # feature has multiple values
        if not min == max:
            for example in range(len(data)):
                data[example][feature_num] = (data[example][feature_num] - min) / (max - min)
        else:
            # remove feature
            for example in range(len(data)):
                del data[example][feature_num]
    return data

# return Symmetric Mean Absolute Percentage Error (SMAPE)
def getDecimalSMAPE(a, p):
    return abs(a - p)/(abs(a)+abs(p))

#--------------------Loss Functions and Paired t Test--------------------#
# When there is a classifier that can return probabilities, use this tester. else use testClassifier
def testProbabilisticClassifier(algorithm, testing_set):
    run_result = []
    for observation_i in range(len(testing_set)):
        obs_result = []
        correct_class = testing_set[observation_i][-1]
        result = algorithm.classify(testing_set[observation_i][:-1])
        predicted = result[0]
        prob = result[1]
        if predicted == correct_class:
            obs_result.append(int(0))
            obs_result.append(1-prob)
        else:
            obs_result.append(int(1))
            obs_result.append(0-prob)
        run_result.append(obs_result)
    return run_result

# 0 if it is classified correctly 1 if it is misclassified for each item in the testing set
def testClassifier(algorithm, testing_set):
    run_result = []
    for observation_i in range(len(testing_set)):
        correct_class = testing_set[observation_i][-1]
        predicted = algorithm.predict(testing_set[observation_i][:-1])
        if predicted == correct_class:
            run_result.append(int(0))
        else:
            run_result.append(int(1))
    return run_result

# absolute error for each item in the testing set
def testRegressor(algorithm, testing_set):
    run_result = []
    for observation_i in range(len(testing_set)):
        y = testing_set[observation_i][-1]
        y_hat = algorithm.predict(testing_set[observation_i][:-1])
        run_result.append(abs(y-y_hat))
    return run_result

# average SMAPE for the testing set
def testAutoencoder(algorithm, testing_set):
    run_SMAPE = 0
    for observation_i in range(len(testing_set)):
        obs_SMAPE = 0
        prediction = algorithm.predict(testing_set[observation_i][:-1])
        for feature in range(len(prediction)):
            obs_SMAPE += getDecimalSMAPE(testing_set[observation_i][feature], prediction[feature])
        #take the average and convert decimal to %
        obs_SMAPE *= (100 / len(prediction))
        run_SMAPE += obs_SMAPE
    run_SMAPE /= len(testing_set)
    return run_SMAPE

# the arrays handed in represent the absolute error in each trial of each run [run][trial]
def compareRegressors(base_missed, other_missed, base_name, other_name):
    base_MSEs = []
    other_MSEs = []
    base_absolute = []
    other_absolute = []
    for run_i in range(10):
        trials = len(base_missed[run_i])
        run_i_base_MSE = 0
        run_i_other_MSE = 0
        run_i_base_absolute = 0
        run_i_other_absolute = 0
        for trial_i in range(trials):
            # since method arguments contain the absolute error for each trial, collect them into absolute error for each run
            run_i_base_absolute += base_missed[run_i][trial_i]
            run_i_other_absolute += other_missed[run_i][trial_i]
            # square the error (MSE)
            run_i_base_MSE += pow(base_missed[run_i][trial_i], 2)
            run_i_other_MSE += pow(other_missed[run_i][trial_i], 2)
        # divide by the number of trials to have the mean (MSE)
        run_i_base_MSE /= trials
        run_i_other_MSE /= trials

        base_MSEs.append(run_i_base_MSE)
        other_MSEs.append(run_i_other_MSE)
        base_absolute.append(run_i_base_absolute)
        other_absolute.append(run_i_other_absolute)

    # for loss functions, calculate the average MSE and absolute error across runs
    base_MSE_avg = 0
    other_MSE_avg = 0
    base_absolute_avg = 0
    other_absolute_avg = 0
    for run_i in range(10):
        base_MSE_avg += base_MSEs[run_i]
        other_MSE_avg += other_MSEs[run_i]
        base_absolute_avg += base_absolute[run_i]
        other_absolute_avg += other_absolute[run_i]
    base_MSE_avg /= len(base_MSEs)
    other_MSE_avg /= len(other_MSEs)
    base_absolute_avg /= len(base_absolute)
    other_absolute_avg /= len(other_absolute)

    print("Comparing Mean Square Error:")
    print(base_name,"MSE:",base_MSE_avg)
    print(other_name, "MSE:",other_MSE_avg)
    print("Comparing Absolute Error")
    print(base_name,"absolute error:", base_absolute_avg)
    print(other_name, "absolute error:", other_absolute_avg)
    # run the paired t test on MSEs with 9 degrees of freedom
    pairedTTest(base_MSEs, other_MSEs, 0.05)

def compareProbabilisticClassifiers(base_missed, other_missed, base_name, other_name):
    # list of MSEs for each run
    base_MSEs = []
    other_MSEs = []
    # list of 0-1 loss for each run
    base_loss_01 = []
    other_loss_01 = []
    for run_i in range(10):
        # number of trials for this run
        trials = len(base_missed[run_i])
        # calculate the MSE for the base algorithm and the other algorithm for trial i
        run_i_base_MSE = 0
        run_i_other_MSE = 0
        run_i_base_loss_01 = 0
        run_i_other_loss_01 = 0
        for trial_i in range(trials):
            run_i_base_MSE += pow(base_missed[run_i][trial_i][1], 2)
            run_i_other_MSE += pow(other_missed[run_i][trial_i][1], 2)
            run_i_base_loss_01 += base_missed[run_i][trial_i][0]
            run_i_other_loss_01 += other_missed[run_i][trial_i][0]
        run_i_base_MSE /= trials
        run_i_other_MSE /= trials
        run_i_base_loss_01 /= trials
        run_i_other_loss_01 /= trials
        # append the MSE for this run to the list of run MSEs
        base_MSEs.append(run_i_base_MSE)
        other_MSEs.append(run_i_other_MSE)
        # append the 0-1 loss for this run to the list of run 0-1 losses
        base_loss_01.append(run_i_base_loss_01)
        other_loss_01.append(run_i_other_loss_01)
    # calculate averages for reporting
    base_MSE_avg = 0
    other_MSE_avg = 0
    base_loss_01_avg = 0
    other_loss_01_avg = 0
    for run_i in range(10):
        base_MSE_avg += base_MSEs[run_i]
        other_MSE_avg += other_MSEs[run_i]
        base_loss_01_avg += base_loss_01[run_i]
        other_loss_01_avg += other_loss_01[run_i]
    base_MSE_avg /= len(base_MSEs)
    other_MSE_avg /= len(other_MSEs)
    base_loss_01_avg /= len(base_loss_01)
    other_loss_01_avg /= len(other_loss_01)

    print("Comparing 0-1 loss:")
    print(base_name, "0-1 Loss:", base_loss_01_avg)
    print(other_name, "0-1 Loss:", other_loss_01_avg)
    print("Comparing MSE:")
    print(base_name, "MSE:", base_MSE_avg)
    print(other_name, "MSE:", other_MSE_avg)
    print("Paired t-test for comparing", base_name, "to", other_name, "using 0-1 loss")

    # run the paired t test on 0-1 loss with 9 degrees of freedom
    pairedTTest(base_loss_01, other_loss_01, 0.05)


# based_missed and other_missed are lists of size 10 containing lists. These inner lists are the result of each of the 10 folds
# with 0 meaning correct classification and 1 meaning incorrect classification
def compareClassifiers(base_missed, other_missed, base_name, other_name):
    # list of MSEs for each run
    base_MSEs = []
    other_MSEs = []
    # list of 0-1 loss for each run
    base_loss_01 = []
    other_loss_01 = []
    for run_i in range(10):
        # number of trials for this run
        trials = len(base_missed[run_i])
        # calculate the MSE for the base algorithm and the other algorithm for trial i
        run_i_base_MSE = 0
        run_i_other_MSE = 0
        run_i_base_loss_01 = 0
        run_i_other_loss_01 = 0
        for trial_i in range(trials):
            run_i_base_MSE += pow(0-base_missed[run_i][trial_i], 2)
            run_i_other_MSE += pow(0-other_missed[run_i][trial_i], 2)
            run_i_base_loss_01 += base_missed[run_i][trial_i]
            run_i_other_loss_01 += other_missed[run_i][trial_i]
        run_i_base_MSE /= trials
        run_i_other_MSE /= trials
        run_i_base_loss_01 /= trials
        run_i_other_loss_01 /= trials
        # append the MSE for this run to the list of run MSEs
        base_MSEs.append(run_i_base_MSE)
        other_MSEs.append(run_i_other_MSE)
        # append the 0-1 loss for this run to the list of run 0-1 losses
        base_loss_01.append(run_i_base_loss_01)
        other_loss_01.append(run_i_other_loss_01)
    base_MSE_avg = 0
    other_MSE_avg = 0
    base_loss_01_avg = 0
    other_loss_01_avg = 0
    for run_i in range(10):
        base_MSE_avg += base_MSEs[run_i]
        other_MSE_avg += other_MSEs[run_i]
        base_loss_01_avg += base_loss_01[run_i]
        other_loss_01_avg += other_loss_01[run_i]
    base_MSE_avg /= len(base_MSEs)
    other_MSE_avg /= len(other_MSEs)
    base_loss_01_avg /= len(base_loss_01)
    other_loss_01_avg /= len(other_loss_01)

    print("Comparing 0-1 loss:")
    print(base_name,"0-1 Loss:", base_loss_01_avg)
    print(other_name,"0-1 Loss:", other_loss_01_avg)
    print("Comparing MSE:")
    print(base_name,"MSE:", base_MSE_avg)
    print(other_name, "MSE:", other_MSE_avg)
    print("Paired t-test for comparing",base_name, "to", other_name, "using 0-1 loss")

    # run the paired t test on 0-1 loss with 9 degrees of freedom
    pairedTTest(base_loss_01, other_loss_01, 0.05)

def pairedTTest(p_a, p_b, alpha):
    p_dif = []
    n = len(p_a)

    # calculate an array of the differences in probabilities
    for i in range(0, n):
        p_dif.append(p_a[i] - p_b[i])

    # calculate average difference
    p_bar = 0.0
    for i in range(0, n):
        p_bar += p_dif[i]
    p_bar /= n

    # calculate t_statistic
    t_statistic = p_bar * pow(n, 0.5)
    t_denom = 0.0
    for i in range(0, n):
        t_denom += pow((p_dif[i] - p_bar), 2)
    t_denom /= (n - 1)
    t_denom = pow(t_denom, 0.5)
    if t_denom != 0:
        t_statistic /= t_denom

    # with 9 df (10-fold t-test)
    print("t_statistic: ")
    print(t_statistic)
    print("\n")
    if alpha >= 0.05:
        # using alpha 0.05
        if abs(t_statistic) > 2.262:
            # reject
            print("With alpha = 0.05, we reject the null\n")
        else:
            # fail to reject
            print("With alpha = 0.05, we fail to reject the null\n")
    elif alpha > 0.01:
        # using alpha 0.02
        if abs(t_statistic) > 2.821:
            # reject
            print("With alpha = 0.02, we reject the null\n")
        else:
            # fail to reject
            print("With alpha = 0.02, we fail to reject the null\n")
    else:
        # using alpha 0.01
        if abs(t_statistic) > 3.25:
            # reject
            print("With alpha = 0.01, we reject the null\n")
        else:
            # fail to reject
            print("With alpha = 0.01, we fail to reject the null\n")
