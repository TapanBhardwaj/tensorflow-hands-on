from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
neta_learning_rate = 0.01
num_iterations = 30000
batch_size = 256

step_display = 1000
examples_to_show = 10

# Network Parameters
num_hidden_units_1 = 256
num_hidden_units_2 = 128
input_dimensions = 784


def define_variables():
    global x_placeholder, weight_encoder_layer1, weight_encoder_layer2, weight_decoder_layer1, weight_decoder_layer2, bias_encoder_layer1, bias_encoder_layer2, bias_decoder_layer1, bias_decoder_layer2
    x_placeholder = tf.placeholder("float", [None, input_dimensions])

    weight_encoder_layer1 = tf.Variable(tf.random_normal([input_dimensions, num_hidden_units_1]))
    weight_encoder_layer2 = tf.Variable(tf.random_normal([num_hidden_units_1, num_hidden_units_2]))
    weight_decoder_layer1 = tf.Variable(tf.random_normal([num_hidden_units_2, num_hidden_units_1]))
    weight_decoder_layer2 = tf.Variable(tf.random_normal([num_hidden_units_1, input_dimensions]))

    bias_encoder_layer1 = tf.Variable(tf.random_normal([num_hidden_units_1]))
    bias_encoder_layer2 = tf.Variable(tf.random_normal([num_hidden_units_2]))
    bias_decoder_layer1 = tf.Variable(tf.random_normal([num_hidden_units_1]))
    bias_decoder_layer2 = tf.Variable(tf.random_normal([input_dimensions]))


# Building the encoder
def build_encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weight_encoder_layer1), bias_encoder_layer1))
    # Encoder Hidden layer with sigmoid activation #2
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weight_encoder_layer2), bias_encoder_layer2))
    return layer2


# Building the decoder
def build_decoder(x):
    # Decoder layer1, activation function sigmoid
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weight_decoder_layer1),
                                  bias_decoder_layer1))
    # Decoder layer2, activation function sigmoid
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weight_decoder_layer2),
                                  bias_decoder_layer2))
    return layer2


# Models construction
def predict_output():
    global encoder_model, decoder_model
    encoder_model = build_encoder(x_placeholder)
    decoder_model = build_decoder(encoder_model)


# Train the network
def train_network():
    global num_of_images, original_image_canvas, reconstructed_image_canvas
    with tf.Session() as session:

        # Running the initializer
        session.run(initialize_variables)

        for iteration in range(1, num_iterations + 1):
            batch_x, _ = mnist.train.next_batch(batch_size)
            _, l = session.run([optimizer, loss], feed_dict={x_placeholder: batch_x})
            # Display logs per step
            if iteration % step_display == 0 or iteration == 1:
                print('Step %i: Loss of Minibatch : %f' % (iteration, l))

        num_of_images = 1
        original_image_canvas = np.empty((28 * num_of_images, 28 * num_of_images))
        reconstructed_image_canvas = np.empty((28 * num_of_images, 28 * num_of_images))
        for iteration in range(num_of_images):
            # MNIST test set
            batch_x, _ = mnist.test.next_batch(num_of_images)
            # Encode and decode the digit image
            image_reconstructed = session.run(decoder_model, feed_dict={x_placeholder: batch_x})

            # Display original images
            for j in range(num_of_images):
                # Draw the original digits
                original_image_canvas[iteration * 28:(iteration + 1) * 28, j * 28:(j + 1) * 28] = \
                    batch_x[j].reshape([28, 28])
            # Display reconstructed images
            for j in range(num_of_images):
                # Draw the reconstructed digits
                reconstructed_image_canvas[iteration * 28:(iteration + 1) * 28, j * 28:(j + 1) * 28] = \
                    image_reconstructed[j].reshape([28, 28])


def print_images():
    print("Original Image")
    plt.figure(figsize=(num_of_images, num_of_images))
    plt.imshow(original_image_canvas, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Image")
    plt.figure(figsize=(num_of_images, num_of_images))
    plt.imshow(reconstructed_image_canvas, origin="upper", cmap="gray")
    plt.show()


define_variables()
predict_output()
# Prediction
y_predicted = decoder_model
# Targets (Labels) are the input data.
y_true = x_placeholder

loss = tf.reduce_mean(tf.pow(y_true - y_predicted, 2))
optimizer = tf.train.RMSPropOptimizer(neta_learning_rate).minimize(loss)

# Initializing the variables
initialize_variables = tf.global_variables_initializer()

train_network()

print_images()
