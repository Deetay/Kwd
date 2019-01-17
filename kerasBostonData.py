import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
from keras import losses
from keras.datasets import boston_housing
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed(1)

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

neural_network_boston = Sequential()
neural_network_boston.add(Dense(10, activation='relu', input_shape=(x_train.shape[1],)))
neural_network_boston.add(Dense(5, activation='relu'))
neural_network_boston.add(Dense(1, activation='linear'))

neural_network_boston.summary()

neural_network_boston.compile(SGD(lr=.003), "mean_squared_error", )

run_hist_1 = neural_network_boston.fit(x_train, y_train, epochs=500, \
                                       validation_data=(x_test, y_test), \
                                       verbose=True, shuffle=False)

print("Training neural network without dropouts..\n")
print("Model evaluation Train data [loss]: ", neural_network_boston.evaluate(x_train, y_train))
print("Model evaluation  Test Data [loss]: ", neural_network_boston.evaluate(x_test, y_test))

neural_network_boston_dropout = Sequential()
neural_network_boston_dropout.add(Dense(10, activation='relu', input_shape=(x_train.shape[1],)))
neural_network_boston_dropout.add(Dropout(0.01))
neural_network_boston_dropout.add(Dense(5, activation='relu'))
neural_network_boston_dropout.add(Dropout(0.01))
neural_network_boston_dropout.add(Dense(1, activation='linear'))

neural_network_boston_dropout.summary()

neural_network_boston_dropout.compile(SGD(lr=.003), "mean_squared_error", )

run_hist_2 = neural_network_boston_dropout.fit(x_train, y_train, epochs=500, \
                                               validation_data=(x_test, y_test), \
                                               verbose=True, shuffle=False)

print("Training neural network with dropouts..\n")
print("Model evaluation Train data [loss]: ", neural_network_boston_dropout.evaluate(x_train, y_train))
print("Model evaluation  Test Data [loss]: ", neural_network_boston_dropout.evaluate(x_test, y_test))

plt.plot(run_hist_1.history["loss"], 'r', marker='.', label="Train Loss")
plt.plot(run_hist_1.history["val_loss"], 'b', marker='.', label="Validation Loss")
plt.title("Train loss and validation error without dropouts")
plt.legend()
plt.grid()
plt.show()
plt.plot(run_hist_2.history["loss"], 'r', marker='.', label="Train Loss")
plt.plot(run_hist_2.history["val_loss"], 'b', marker='.', label="Validation Loss")
plt.title("Train loss and validation error with dropouts")
plt.legend()
plt.grid()
plt.show()
