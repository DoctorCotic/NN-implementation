import numpy
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

begin_time = time.time()
numpy.random.seed(42)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# marks to category
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# create model
model = Sequential()

# add range of network
model.add(Dense(800, input_dim=784, activation="relu", kernel_initializer="normal")) #enter layer
model.add(Dense(10, activation="softmax", kernel_initializer="normal"))              #exit layer
# add hidden layers
model.add(Dense(800, input_dim=784, activation="relu", kernel_initializer="normal"))
model.add(Dense(600, activation="relu", kernel_initializer="normal"))
model.add(Dense(10, activation="softmax", kernel_initializer="normal"))

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

#SGD -  stochastic gradient descent

print(model.summary())
model.fit(X_train, Y_train, batch_size=200, epochs=12, validation_split=0.2, verbose=2)
#20% of data use for test sample

# evaluate the quality of the network training on the test data
scores = model.evaluate(X_test, Y_test, verbose=0)
print("The accuracy of the model on test data %.2f%%" % (scores[1]*100))

model_json = model.to_json()
json_file = open("mnist_model.json","w")
json_file.write(model_json)
json_file.close()

end_time = time.time()
t = end_time - begin_time
print(t)
