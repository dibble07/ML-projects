# Notes
# Neural network code to optimise training of MNIST digit dataset
# and extract number from webcam and play an audio recording of that sequence of numbers

print("""
To do:
	save and load trained neural networks
	optimise other parameters - batch size, number and size of layers (read article on this)
	change activation function
	change layer type
	consider uncertainty in final results
	plot maximum activation images for each layer
	consider max pooling and regularisation
	read from webcam, split image, resample, process, play audio
	train other datasets - see available ones in keras
""")

# Import libraries
import keras
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd
import time
matplotlib.use('TkAgg')
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', None)

# User defined functions

def neural_network(arch_layers):

	# Define network architecture
	model = keras.models.Sequential()
	if arch_layers[0][0] is "dense":
		print(x_train.shape[1:])
		model.add(keras.layers.Dense(units=arch_layers[0][1], activation=arch_layers[0][2], input_shape=x_train.shape[1:]))
	# model.add(keras.layers.Flatten())
	for arch_layer in arch_layers[1:]:
		if arch_layer[0] is "dense":
			model.add(keras.layers.Dense(units=arch_layer[1], activation=arch_layer[2]))

	# Train weights and biases
	model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()
	callbacks=[keras.callbacks.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, min_delta=min_delta, patience=patience)]
	start = time.process_time()
	model_fit = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_valid, y_valid), callbacks=callbacks)
	time_run=time.process_time() - start
	early_stop = len(model_fit.epoch) < model_fit.params["epochs"]
	
	return early_stop, time_run, model, model_fit

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_valid = x_test[0:5000]
y_valid = y_test[0:5000]
x_test = x_test[5000:]
y_test = y_test[5000:]
x_train = np.concatenate((x_train, np.random.randint(low=0, high=255, size=(6000, 28, 28))), axis=0)
y_train = np.concatenate((y_train, np.tile(-1, 6000)), axis=0)
x_valid = np.concatenate((x_valid, np.random.randint(low=0, high=255, size=(500, 28, 28))), axis=0)
y_valid = np.concatenate((y_valid, np.tile(-1, 500)), axis=0)
x_test = np.concatenate((x_test, np.random.randint(low=0, high=255, size=(500, 28, 28))), axis=0)
y_test = np.concatenate((y_test, np.tile(-1, 500)), axis=0)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1]*x_valid.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
num_classes = len(set(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Network hyper parameters
batch_size=64
min_delta=.001
patience=10
epochs=1000
layer_params=[
# ['dense', 1024 , 'sigmoid'],
['dense', 512 , 'sigmoid'],
['dense', num_classes, 'softmax']
]

# Run model
early_stop, time_fit, model, model_fit = neural_network(layer_params)

# Save architecture and fit performance
arch_dict = {
"layer_type":[",".join([a[0] for a in layer_params])],
"layer_size":[",".join([f"{a[1]}" for a in layer_params])],
"layer_activ":[",".join([a[2] for a in layer_params])],
"batch_size":[batch_size],
"min_delta":[min_delta],
"patience":[patience],
"early_stop":[early_stop],
"time_fit":[time_fit],
"valid_acc":[model_fit.history['val_accuracy'][-1]]
}
arch_df = pd.DataFrame(arch_dict)
print("Current result:")
print(arch_df)
filename="arch_hist.csv"
if os.path.isfile(filename):
	arch_hist_df = pd.read_csv(filename)
	arch_hist_df.sort_values(by=["valid_acc"], inplace=True, ascending=False)
	print("Previous result:")
	print(arch_hist_df)
	arch_hist_df=arch_hist_df.append(arch_df)
else:
	arch_hist_df=arch_df
arch_hist_df.to_csv(filename, index = False)

# Plot results of training
plt.plot(model_fit.history['accuracy'])
plt.plot(model_fit.history['val_accuracy'])
if early_stop:
	plt.title('model accuracy')
else:
	plt.title('model accuracy\nEARLY STOP NOT ACHIEVED!')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

# Save model for future use
model.save("model.h5")