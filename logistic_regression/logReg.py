from numpy import *
import matplotlib.pyplot as plt

def sigmoid(inX):
    return 1.0/(1 + exp(-inX))

def trainLogReg(xtrain, ytrain, params):
    numSample, numFeature = shape(xtrain)
    alpha = params['alpha']
    maxIter = params['maxIter']
    weights = ones((numFeature, 1))
    
    for i in range(maxIter):
        if params['optimizeType'] == 'gradDescent': #gradient descent
            output = sigmoid(xtrain * weights)
            error = ytrain - output
            weights = weights + alpha * xtrain.transpose() * error
        elif params['optimizeType'] == 'stocGradDescent': #stochastic gradient descent
            for j in range(numSample):
                output = sigmoid(xtrain[j,:] * weights)
                error = ytrain[j, 0] - output
                weights = weights + alpha * xtrain[j, :].transpose() * error
        elif params['optimizeType'] == 'smoothStocGradDescent': #smooth stochastic gradient descent
            dataIndex = range(numSample)
            for j in range(numSample):
                alpha = 40/(1.0 + i + j) + 0.2
                randIndex = int(random.uniform(0, len(dataIndex)))
                output = sigmoid(xtrain[randIndex, :] * weights)
                error = ytrain[randIndex, 0] - output
                weights = weights + alpha * xtrain[randIndex,:].transpose() * error
                del(dataIndex[randIndex]) #after one iteration, delete the sample
        else:
            raise NameError('Optimize type not valid!')
            
    return weights


def testLogReg(weights, xtest, ytest):
    numSample, numFeature = shape(xtest)
    correct = 0
    for i in xrange(numSample):
        predict = sigmoid(xtest[i,:] * weights)[0, 0] > 0.5
        if predict == ytest[i, 0]:
            correct += 1
    accuracy = float(correct) / numSample
    return accuracy


def plotLogReg(weights, xtrain, ytrain):
    numSample, numFeature = shape(xtrain)
    #only works for 2D array
    #draw all dots
    for i in range(numSample):
        if (ytrain[i] == 0):
            plt.plot(xtrain[i, 1], xtrain[i, 2], 'or')
        elif (ytrain[i] == 1):
            plt.plot(xtrain[i, 1], xtrain[i, 2], 'ob')
            
    min_x = min(xtrain[:, 1])[0, 0]
    max_x = max(xtrain[:, 1])[0, 0]
    weights = weights.getA() #convert mat to array
    y_min = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min, y_max], '-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
