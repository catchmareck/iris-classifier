import numpy as np
import random


def reLU(x):
    return np.maximum(0, x)


def softplus_deriv(x):
    return sigmoid(x)


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def update_weights(weights, deltas):
    new_weights = np.array(weights)

    for x in range(len(weights[0])):
        new_weights[:,x] += learning_rate * deltas[x] * bias

    return new_weights


def get_target(text_target):
    if text_target.strip() == 'Iris-virginica':
        return np.array([1, 0, 0])
    elif text_target.strip() == 'Iris-setosa':
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])


learning_rate = 0.1
bias = 1

data = open('iris.data').readlines()

dataset = []
for line in data:
    linesplit = line.split(',')
    dataset.append(linesplit)

random.shuffle(dataset)
trainset = dataset[:120]
testset = dataset[120:]

n_input = 4
n_hidden = 4
n_output = 3


W1 = np.array([np.random.dirichlet(np.ones(n_hidden), size=1)[0] for i in range(n_input)])
W2 = np.array([np.random.dirichlet(np.ones(n_output), size=1)[0] for i in range(n_hidden)])

for epoch in range(100):

    print('epoch %d' % epoch)
    for sample in trainset:

        INPUT = np.array([float(sample[i]) for i in range(n_input)])
        TARGET = get_target(sample[n_input])

        H = np.array([tanh(INPUT.dot(W1[:,x])) for x in range(n_hidden)])
        O = np.array([sigmoid(H.dot(W2[:,x])) for x in range(n_output)])

        O_e = TARGET - O
        H_e = O_e.dot(W2.T)

        O_d = np.array([(sigmoid_deriv(H.dot(W2[:,x])) * O_e[x]) for x in range(n_output)])
        H_d = np.array([(tanh_deriv(INPUT.dot(W1[:,x])) * H_e[x]) for x in range(n_hidden)])

        W2 = update_weights(W2, O_d)
        W1 = update_weights(W1, H_d)


# testing
def classify(testset):
    for sample in testset:
        I = np.array([float(sample[i]) for i in range(n_input)])
        TARGET = get_target(sample[n_input])
        H = np.array([tanh(I.dot(W1[:,x])) for x in range(n_hidden)])
        O = np.array([sigmoid(H.dot(W2[:,x])) for x in range(n_output)])

        print('for this sample', I, 'it should be', TARGET, 'but got', O, 'so', np.argmax(TARGET) == np.argmax(O))


classify(testset)
