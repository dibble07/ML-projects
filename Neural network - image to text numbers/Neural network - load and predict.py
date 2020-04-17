# Import libraries
import keras
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

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
x_test_orig = x_test
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
num_classes = len(set(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test_orig = y_test
y_test = keras.utils.to_categorical(y_test, num_classes)

# Load trained neural network
model = keras.models.load_model('model.h5')
model.summary()


eval_test=x_test_orig[0,:,:]
print(eval_test.shape)
eval_test_flat = eval_test.reshape(-1,eval_test.shape[0]*eval_test.shape[1])
print(eval_test_flat.shape)
print(eval_test_flat)
hey = model.predict(eval_test_flat)
print(hey)
print(np.argmax(hey))
figure, axes = plt.subplots(nrows=1, ncols=1)
axes.imshow(eval_test, cmap='gray')
figure.tight_layout()
plt.show()