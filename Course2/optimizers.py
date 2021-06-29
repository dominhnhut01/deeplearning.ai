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
