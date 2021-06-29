import numpy as np

def random_minibatches(X, Y, batch_size = 64):

	m_training_examples = Y.shape[1]
	#Shuffle X in place with Y
	permutation = np.random.permutation(m_training_examples)
	shuffle_X = X[:,permutation]
	shuffle_Y = Y[:, permutation]

	X_minibatches = []
	Y_minibatches = []
	for i in range(0, m_training_examples, batch_size):
		X_minibatches.append(shuffle_X[:,i : i + batch_size])
		Y_minibatches.append(shuffle_Y[:,i : i + batch_size])

	return X_minibatches, Y_minibatches

def initialize_gd_with_momentum_params(layers_dims):
    net_depth = len(layers_dims)

    v = {}
    for l in range(1, net_depth + 1):
        v["vdw" + str(l)] = 0
        v["vdb" + str(l)] = 0
    return v

def initialize_adam_params(layers_dims):
    net_depth = len(layers_dims)

    v = {}
    s= {}

    for l in range(1, net_depth + 1):
        v["vdw" + str(l)] = 0
        v["vdb" + str(l)] = 0
        s["sdw" + str(l)] = 0
        s["sdb" + str(l)] = 0

    return v, s
def update_params_with_momentum(params, grads, v, learning_rate, beta = 0.9):
    net_depth = len(params) // 2

    for l in range(1, net_depth + 1):
        v["vdw" + str(l)] = beta * v["vdw" + str(l)] + (1-beta) * grads["dw" + str(l)]
        v["vdb" + str(l)] = beta * v["vdb" + str(l)] + (1-beta) * grads["db" + str(l)]

        params["w" + str(l)] = params["w" + str(l)] - learning_rate * v["vdw" + str(l)]
        params["b" + str(l)] = params["b" + str(l)] - learning_rate * v["vdb" + str(l)]
    return params, v

def update_params_adam_optimizer(params, grads, cur_iteration, v, s, learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    net_depth = len(params) // 2

    for l in range(1, net_depth+1):

        #Exponentially weighted avg
        v["vdw" + str(l)] = beta1 * v["vdw" + str(l)] + (1-beta1) * grads["dw" + str(l)]
        v["vdb" + str(l)] = beta1 * v["vdb" + str(l)] + (1-beta1) * grads["db" + str(l)]

        #Bias correction for vdw and vdb
        vdw_corrected = v["vdw" + str(l)]/(1-pow(beta1, cur_iteration))
        vdb_corrected = v["vdb" + str(l)]/(1-pow(beta1, cur_iteration))

        #RMS
        s["sdw" + str(l)] = beta2 * s["sdw" + str(l)] + (1 - beta2) * np.square(grads["dw" + str(l)])
        s["sdb" + str(l)] = beta2 * s["sdb" + str(l)] + (1 - beta2) * np.square(grads["db" + str(l)])

        #Bias correction for sdw and sdb
        sdw_corrected = s["sdw" + str(l)]/(1-pow(beta2, cur_iteration))
        sdb_corrected = s["sdb" + str(l)]/(1-pow(beta2, cur_iteration))

        params["w" + str(l)] = params["w" + str(l)] - learning_rate * vdw_corrected / (np.sqrt(sdw_corrected) + epsilon)
        params["b" + str(l)] = params["b" + str(l)] - learning_rate * vdb_corrected / (np.sqrt(sdb_corrected) + epsilon)

    return params, v, s
