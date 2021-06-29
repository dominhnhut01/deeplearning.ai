from backbone import *
from regularizations import *
import numpy as np

if __name__ == '__main__':
	train_x, train_y = load_dataset()
	print(train_x.shape)
	print(train_y.shape)
	m_train = train_x.shape[0]
	num_px = train_x.shape[1]

	# train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
	# test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

	# Standardize data to have feature values between 0 and 1.
	# train_x = train_x_flatten/255.0
	# test_x = test_x_flatten/255.0

	# layers_dims = (12288,20,7,5,1)
	layers_dims = [train_x.shape[0], 20, 3, 1]
	drop_out_keep_prob = [1,0.8,0.8,1]
	regularization_param = 0.1

	epoch_num = 300
	batch_size = 64

	params, costs_train = model(train_x, train_y, layers_dims = layers_dims, drop_out_keep_prob = drop_out_keep_prob, epoch_num = epoch_num, batch_size = batch_size, learning_rate = 0.0075, optimizer = "adam", regularization_param = regularization_param, beta1 = 0.9, beta2 = 0.999)
	plt.plot([epoch for epoch in range(0, epoch_num, 100)],costs_train)

	plt.show()

	train_prediction = predict(train_x, params, layers_dims)

	train_accuracy = computeAccuracy(train_prediction, train_y)

	print("Accuracy on training dataset is: {}".format(train_accuracy))

	plt.title("Model results")

	axes = plt.gca()
	axes.set_xlim([-1.5,2.5])
	axes.set_ylim([-1,1.5])
	plot_decision_boundary(train_x, train_y, params, layers_dims)

	plt.show()
