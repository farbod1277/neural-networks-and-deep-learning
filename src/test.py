import os
os.chdir("./src")

import mnist_loader
import network

sizes = [784, 30, 10]
epochs = 30
mini_batch_size = 10
eta = 3.0

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network(sizes)
net.SGD(training_data, epochs, mini_batch_size,eta, test_data=test_data)