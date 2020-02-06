import tensorflow as tf
import numpy as np


# helper function which helps turn strings into numbers
def label_encode(label):
    val = []
    if label == "Iris-setosa":
        val = [1, 0, 0]
    elif label == "Iris-versicolor":
        val = [0, 1, 0]
    elif label == "Iris-virginica":
        val = [0, 0, 1]
    return val


# helper function which encodes all data using label_encode() helper line by line
def data_encode(file):
    X = []
    Y = []
    train_file = open(file, 'r')
    for line in train_file.read().strip().split('\n'):
        line = line.split(',')
        X.append([line[0], line[1], line[2], line[3]])
        Y.append(label_encode(line[4]))
    return X, Y


# model parameters
learning_rate = 0.01
training_epochs = 5000
display_steps = 100

n_input = 4
n_hidden = 10
n_output = 3

# TF placeholders which will represent X and Y input training data
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

# fill weights with random values form Normal Distribution (Gaussian)
weights = {
    "hidden": tf.Variable(tf.random_normal([n_input, n_hidden])),
    "output": tf.Variable(tf.random_normal([n_hidden, n_output])),
}

# fill bias with random values form Normal Distribution (Gaussian)
bias = {
    "hidden": tf.Variable(tf.random_normal([n_hidden])),
    "output": tf.Variable(tf.random_normal([n_output])),
}


# function which creates a model. Add one hidden layer and one output layer
def model(X, weights, bias):
    layer1 = tf.add(tf.matmul(X, weights["hidden"]), bias["hidden"])
    layer1 = tf.nn.relu(layer1)  # reLU activation function

    output_layer = tf.matmul(layer1, weights["output"]) + bias["output"]
    return output_layer


# get test and train sets
train_X, train_Y = data_encode("iris.train")
test_X, test_Y = data_encode("iris.test")

# create a model using our input values, weights and bias vectors
pred = model(X, weights, bias)

# cost function (error function)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
# AdamOptimizer is a specialized adaptive optimizer which is a variation of SGD
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# initialize TF variables
init = tf.global_variables_initializer()

# start TF session
with tf.Session() as sess:
    sess.run(init)

    # run all epochs
    for epochs in range(training_epochs):
        # evaluate the model on our train_X and train_Y sets using defined AdamOptimizer and cost function
        _, c = sess.run([optimizer, cost], feed_dict={X: train_X, Y: train_Y})
        # helper if for displaying progress
        if(epochs + 1) % display_steps == 0:
            print("Epoch:", epochs+1, "Cost:", c)

    print("Optimization Finished")

    # helper prints for display purposes
    print(train_X)
    print(np.argmax(train_Y, 1))

    # run our model through test set
    test_result = sess.run(pred, feed_dict={X: train_X})
    print(np.argmax(test_result, 1))

    # get model prediction and calculate its accuracy
    correct_prediction = tf.equal(tf.argmax(test_result, 1), tf.argmax(train_Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("accuracy:", accuracy.eval({X: test_X, Y: test_Y}))
