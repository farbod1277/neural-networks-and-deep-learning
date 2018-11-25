import os
os.chdir("./src")

import mnist_loader
import network

sizes = [1,1]
epochs = 20
mini_batch_size = 1
eta = 0.15

#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network(sizes)
print net.SGD([(1,0),]*20, epochs, mini_batch_size,eta)

