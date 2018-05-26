from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
neta_learning_rate = 0.001
training_iterations = 15
batchsize = 100
display_step = 1

# Network Parameters
num_hidden_layer1_units = 256
num_hidden_layer2_units = 256
num_of_inputs = 784
num_of_labels = 10


def initianlize_variables():
    global x_placeholder, y_placeholder, w1, w2, wout, b1, b2, bout
    x_placeholder = tf.placeholder("float", [None, num_of_inputs])
    y_placeholder = tf.placeholder("float", [None, num_of_labels])

    w1 = tf.Variable(tf.random_normal([num_of_inputs, num_hidden_layer1_units]))
    w2 = tf.Variable(tf.random_normal([num_hidden_layer1_units, num_hidden_layer2_units]))
    wout = tf.Variable(tf.random_normal([num_hidden_layer2_units, num_of_labels]))

    b1 = tf.Variable(tf.random_normal([num_hidden_layer1_units]))
    b2 = tf.Variable(tf.random_normal([num_hidden_layer2_units]))
    bout = tf.Variable(tf.random_normal([num_of_labels]))


# Create model
def multi_layer_perceptron(x):
    # Hidden layer1 with 256 neurons
    layer_1 = tf.add(tf.matmul(x, w1), b1)
    # Hidden layer2 with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
    # Output layer with one neuron for each class
    out_layer = tf.matmul(layer_2, wout) + bout
    return out_layer


def run_model_evaluate_accuracy():
    with tf.Session() as session:
        session.run(initialize_variables)

        # Training cycle
        for iteration in range(training_iterations):
            average_cost = 0.
            num_of_batches = int(mnist.train.num_examples / batchsize)
            # Loop over all batches
            for i in range(num_of_batches):
                batch_x, batch_y = mnist.train.next_batch(batchsize)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = session.run([train_op, loss_op], feed_dict={x_placeholder: batch_x,
                                                                   y_placeholder: batch_y})
                # Compute average loss
                average_cost += c / num_of_batches
            # Display logs per epoch step
            if iteration % display_step == 0:
                print("Iteration:", '%04d' % (iteration + 1), "cost={:.9f}".format(average_cost))
        print("Optimization Complete!")

        # Test model
        pred_labels = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred_labels, 1), tf.argmax(y_placeholder, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x_placeholder: mnist.test.images, y_placeholder: mnist.test.labels}))


# Initializing variables
initianlize_variables()

# Constructing the model
logits = multi_layer_perceptron(x_placeholder)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=y_placeholder))
optimizer = tf.train.AdamOptimizer(learning_rate=neta_learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
initialize_variables = tf.global_variables_initializer()

run_model_evaluate_accuracy()
