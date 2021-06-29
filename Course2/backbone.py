import numpy as np
import matplotlib.pyplot as plt
import h5py
from regularizations import *
import scipy.io
from optimizers import *
import sklearn
import sklearn.datasets

def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    plt.show()

    return train_X, train_Y

def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);
    plt.show()

    return train_X, train_Y, test_X, test_Y

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def plot_decision_boundary(X, Y, params, layers_dims):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    coords = np.c_[xx. ravel(), yy.ravel()]
    predictions= predict(coords.T, params, layers_dims)

    predictions = np.array(predictions)
    predictions = predictions.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, predictions, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)
    plt.show()

def sigmoid(mat):
    return 1/(1+np.exp(-mat))

def relu(mat):
    return np.maximum(0.0, mat)

def reluBackward(dA, Z):
    Z[Z<=0] = 0
    Z[Z>0] = 1
    dZ = dA * Z
    return dZ
def sigmoidBackward(dA, A):
    return dA * A * (1-A)

def initializeParams(layers_dims):
    """
    Initialize the weights and bias for the neural network

    Argumemts:
    layers_dims -- list containig dimensions of the layers

    Returns:
    params -- dictionary containing value of weight and bias for all layers
    """

    L = len(layers_dims)
    params = {}
    for l in range(1, L):
        params["w" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) / np.sqrt(layers_dims[l-1])
        params["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return params

def linearForward(A_prev, w, b):
    """
    Compute the Z at current layer

    Arguments:
    A_prev -- A value from previous layer
    w -- weight at current layer
    b --  bias at current layer

    Returns:
    Z -- Z value of current layer
    linearCache -- tuple containing (A,w,b)
    """

    Z = np.dot(w, A_prev) + b

    return Z

def linearActivationForward(Z, activation):
    """
    Compute the A at current layer

    Arguments:
    Z -- Z value at current layer
    activation -- a String declaring the name of desired activation function

    Returns:
    A -- A value at current layer
    """

    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    return A
def modelForwardPropagation(X, params, layers_dims, drop_out_keep_prob):
    """
    Operating the forward propagation step through L layer in one iteration

    Arguments:
    X -- data, numpy array (height * width * 3, m_training_examples)
    Y -- true label of the X data, numpy array (1, m_training_examples)
    params -- the dictionary storing the value of weights and biases
    layer_dims -- the dimension of the layers

    Returns:
    AL -- the A value at the Lth layer
    caches -- the list of all the (A_prev, Z, A, w,b) value at each layer
    """
    L = len(layers_dims)
    caches = []
    A_prev = X
    for l in range(1, L-1):
        current_w = params["w" + str(l)]
        current_b = params["b" + str(l)]

        Z = linearForward(A_prev, current_w, current_b)
        A = linearActivationForward(Z, activation = "relu")
        D = np.random.rand(A.shape[0], A.shape[1]) < drop_out_keep_prob[l - 1]
        A = A * D / drop_out_keep_prob[l - 1]

        cache = (A_prev, Z, A, current_w, current_b, D)
        A_prev = A
        caches.append(cache)

    current_w = params["w" + str(L-1)]
    current_b = params["b" + str(L-1)]

    Z = linearForward(A_prev, current_w, current_b)
    AL= linearActivationForward(Z, activation = "sigmoid")
    D = np.random.rand(AL.shape[0], AL.shape[1]) < drop_out_keep_prob[L-2]
    cache = (A_prev, Z, AL, current_w, current_b, D)	#Because we do not drop out last layer
    caches.append(cache)
    return AL, caches

def computeCost (AL, Y, params, regularization_param):
    """
    Compute the cost function

    Arguments:
    AL -- the A value at the Lth layer
    Y -- true label of the X data, numpy array (1, m_training_examples)

    Returns:
    cost -- the value of cost function
    """

    cost = -1 * (np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1 - AL).T))
    cost = np.squeeze(cost) + L2_norm_cost(params, regularization_param)

    return cost

def linearActivationBackward(dA, cache ,activation):
    """
    Compute the derivative of cost with respect to Z at the current layer (only for layer 1 to layer L-1)

    Arguments:
    dA -- derivative of cost with respect to A at the current layer
    cache -- a tuple (A_prev, Z, A, w,b) value at the current layer
    activation -- a String declaring the name of the current activation function

    Returns:
    dA_prev -- derivative of cost with respect to A at the previous layer
    """
    (A_prev, Z, A, w, b, D) = cache
    if activation == "sigmoid":
        dZ = sigmoidBackward(dA, A)
    if activation == "relu":
        dZ = reluBackward(dA, Z)
    return dZ

def linearBackward(dZ, cache, regularization_param):
    """
    Compute the derivative of cost with respect to A at the previous layer, derivative of weights and biases at the current layer

    Arguments:
    dZ -- derivative of cost with respect to Z at current layer
    cache -- a tuple (A_prev, Z, A, w,b) value at the current layer

    Returns:
    dA_prev -- derivative of cost with respect to A at the previous layer
    dw -- derivative of the cost with respect to weight at the current layer
    db -- derivative of the cost with respect to bias at the current layer
    """
    (A_prev, Z, A, w, b, D) = cache
    m = Z.shape[1]

    dA_prev = np.dot(w.T, dZ)
    dw = 1/m * np.dot(dZ, A_prev.T) + L2_norm_backprop(m, w, regularization_param)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)

    return dA_prev, dw, db


def modelBackwardPropagation(Y, AL, caches, layers_dims, regularization_param, drop_out_keep_prob):
    """
    Operating the back propagation step through L layers of each iteration

    Arguments:
    Y -- true label of the X data, numpy array (1, m_training_examples)
    AL -- the A value at the Lth layer at the previous forward propagation step
    caches -- the list of all the (A_prev, Z, A, w, b, D) value at each layer
    layers_dims -- the dimension of the layers

    Returns:
    grads -- the dictionary storing gradients dw, db, dA of each layer
    """
    L = len(layers_dims)
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    dZL = linearActivationBackward(dAL, caches[L-2], activation = "sigmoid")
    dA_prev, dw, db = linearBackward(dZL, caches[L-2], regularization_param)
    grads = {}
    grads["dA" + str(L-1)] = dAL
    grads["dw" + str(L-1)] = dw
    grads["db" + str(L-1)] = db

    for l in range(L-2, 0, -1):
        dA = dA_prev
        dA = dA * caches[l-1][5] / drop_out_keep_prob[l] #Drop out weight
        dZ = linearActivationBackward(dA, caches[l-1], activation = "relu")
        dA_prev, dw, db = linearBackward(dZ, caches[l-1], regularization_param)
        grads["dA" + str(l)] = dA
        grads["dw" + str(l)] = dw
        grads["db" + str(l)] = db
    return grads
def updateParams(params, grads, learning_rate):
    """
    Update parameters after each iteration

    Arguments:
    params -- the dictionary storing the value of weights and biases
    grads -- the dictionary storing gradients dw, db, dA of each layer
    learning_rate -- learning rate

    Returns:
    params -- updated params variables
    """
    L = int(len(params) / 2)

    for l in range(1, L+1):
        params["w" + str(l)] = params["w" + str(l)] - grads["dw" + str(l)] * learning_rate
        params["b" + str(l)] = params["b" + str(l)] - grads["db" + str(l)] * learning_rate

    return params


def predict(X, params, layers_dims):
    """
    Use to predict the label with the input parameters

    Argumemts:
    X -- data, numpy array (height * width * 3, m_training_examples)
    params -- the dictionary storing the value of weights and biases

    Returns:
    prediction -- the predicted label for X
    """
    drop_out_keep_prob = []
    for i in range(len(params) // 2): drop_out_keep_prob.append(1)
    prediction, cache = modelForwardPropagation(X, params, layers_dims, drop_out_keep_prob)
    prediction = (prediction > 0.5)
    return prediction

def computeAccuracy(prediction, Y):
    """
    Compute accuracy

    Argumemts:
    prediction -- the predicted label for X
    Y -- true label of the X data, numpy array (1, m_training_examples)

    Returns:
    accuracy -- accuracy
    """

    prediction = np.squeeze(prediction)
    Y = np.squeeze(Y)

    count = 0

    for i in range(len(prediction)):
        if prediction[i] == Y[i]:
            count+=1
    accuracy = count*1.0/len(Y)

    return accuracy

def model(X,Y, layers_dims, drop_out_keep_prob, epoch_num = 1000, batch_size = 64, learning_rate=0.0075, optimizer = "gd", regularization_param = 0, beta1 = 0.9, beta2 = 0.999):
    """
    Train the model to have appropriate weight and bias for a dataset

    Argumemts:
    X -- data, numpy array (height * width * 3, m_training_examples)
    Y -- true label of the X data, numpy array (1, m_training_examples)
    layer_dims -- the dimension of the layers
    epoch_num -- epoch_num
    learning_rate -- learning rate

    Returns:
    params -- appropriate parameters to predict
    costs -- costs of each 100 iterations
    """

    params = initialize_params_he(layers_dims)
    costs_train = []

    m_training_examples = Y.shape[1]

    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_gd_with_momentum_params(layers_dims)
    elif optimizer == "adam":
        v,s = initialize_adam_params(layers_dims)
        iteration = 0 #for Adam

    for i in range(epoch_num):
        X_minibatches, Y_minibatches = random_minibatches(X, Y, batch_size)
        cost_train_sum = 0
        cost_test_sum = 0
        for minibatch_num in range(len(X_minibatches)):
            AL, caches = modelForwardPropagation(X_minibatches[minibatch_num], params, layers_dims, drop_out_keep_prob)
            caches_size = []
            for cache in caches:
                cache_size = [x.shape for x in cache]
                caches_size.append(cache_size)
            cost_train_sum += computeCost(AL, Y_minibatches[minibatch_num], params, regularization_param)
            grads = modelBackwardPropagation(Y_minibatches[minibatch_num], AL, caches, layers_dims, regularization_param, drop_out_keep_prob)

            if optimizer == "gd":
                params = updateParams(params, grads, learning_rate)
            elif optimizer == "momentum":
                params, v = update_params_with_momentum(params, grads, v, learning_rate, beta1)
            elif optimizer == "adam":
                iteration += 1
                params, v, s = update_params_adam_optimizer(params, grads, iteration, v, s, learning_rate, beta1, beta2)

        cost_train_avg = cost_train_sum / m_training_examples
        if i % 100 == 0:
            costs_train.append(cost_train_avg)
            print("Cost of {}th epoch: {}".format(i+1, cost_train_avg))
    return params, costs_train
