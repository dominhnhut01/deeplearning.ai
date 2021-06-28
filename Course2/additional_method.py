import numpy as np

def initialize_params_he(layers_dims):
	"""
	Initialize the weights and bias for the neural network according to He method

	Argumemts:
	layers_dims -- list containig dimensions of the layers

	Returns:
	params -- dictionary containing value of weight and bias for all layers
	"""

	np.random.seed(3)
	params = {}

	net_depth = len(layers_dims) - 1    #number of layers

	for l in range(1, net_depth+1):
		params["w" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
		params["b" + str(l)] = np.zeros((layers_dims[l],1))

	return params

def L2_norm_cost(m_training_examples, params, regularization_param = 0):
	if regularization_param == 0:
		return 0

	net_depth = int(len(params)/2)
	L2_norm = 0
	for l in range (1, net_depth+1):
		L2_norm += np.square(np.linalg.norm(params["w" + str(l)], ord = "fro"))

	L2_cost = regularization_param / (2 * m_training_examples) * L2_norm

	return L2_cost

def L2_norm_backprop(m_training_examples, cur_w, regularization_param):
	if regularization_param == 0:
		return 0
	return regularization_param / m_training_examples * cur_w
