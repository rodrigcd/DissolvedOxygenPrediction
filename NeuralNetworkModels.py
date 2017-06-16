import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn


class LstmLayer(object):
    """Long Short Term Memory Layer"""
    def __init__(self, input_tensor, hidden_units, time_steps, layer_name):

        self.input_tensor = input_tensor
        self.time_steps = time_steps
        self.layer_name = layer_name
        self.hidden_units = hidden_units

        with tf.variable_scope(layer_name):
            self.lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
            self.state = self.lstm.zero_state(input_tensor.get_shape()[0], dtype='float32')
            self.output_tensor, self.state = self.lstm(self.input_tensor, self.state)


class FullyConnectedLayer(object):
    """fully connected layer"""
    def __init__(self, input_tensor, weights_shape, layer_name):

        self.input_tensor = input_tensor
        self.layer_name = layer_name
        self.weights_shape = weights_shape

        with tf.variable_scope(layer_name):
            self.weights = tf.get_variable("weights", self.weights_shape,
                                           initializer=tf.contrib.layers.xavier_initializer())
            self.biases = tf.get_variable("biases", [self.weights_shape[1]],
                                          initializer=tf.constant_initializer(0.0))
            self.mult_out = tf.matmul(self.input_tensor, self.weights)
            self.output_tensor = tf.nn.sigmoid(self.mult_out + self.biases)


class ForecastingNetwork(object):
    """Convolutional neural network"""

    def __init__(self, **kwargs):
        # TODO: Get performance and stuffs
        self.sequence_length = kwargs["sequence_length"]

        self.database = []
        self.sess = tf.Session()
        self.build_model()
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)
        self.optimizer = tf.train.GradientDescentOptimizer(5e-4)
        self.train_all_params = self.optimizer.minimize(self.loss)
        self.correct_predictions = tf.equal(tf.argmax(self.model_output, 1),
                                            tf.argmax(self.target, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_predictions, tf.float32), name='accuracy')

    def build_model(self):

        self.model_input = tf.placeholder(tf.float32)
        self.target = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)

        self.lstm_layer_1 = LstmLayer(input_tensor=self.model_input,
                                     hidden_units=5,
                                     time_steps=5,
                                     layer_name="lstm_layer1")

        self.fc_layer_2 = FullyConnectedLayer(input_tensor=self.lstm_layer_1.output_tensor,
                                             weights_shape=[5, 1],
                                             layer_name="fc_layer2")

        self.model_output = self.fc_layer_2.output_tensor

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                self.model_output,
                self.target,
                name='loss')
            )

    def train_iterations(self, n_iterations):

        train_step = self.train_all_params
        accuracy_values = list()
        #test_accuracy = list()

        for iteration in range(n_iterations):
            data, labels = self.database.next_batch()
            _ = self.sess.run((train_step),feed_dict={
                self.model_input: data,
                self.target: labels,
                self.keep_prob: 0.5
            })
            if iteration%50 == 0:
                train_accuracy = self.accuracy.eval(
                    feed_dict={
                        self.model_input: data,
                        self.target: labels,
                        self.keep_prob: 0.5# to be fair
                        }, session=self.sess
                    )
                print("train accuracy = " + str(train_accuracy))
                accuracy_values.append(train_accuracy)
                #test_accuracy.append(self.evaluate())
        return accuracy_values#, test_accuracy

    def evaluate(self, data=[], use_test=False):
        if not data:
            if use_test:
                batches = self.light_curves.test_set
            else:
                batches = self.light_curves.validation_set
        accuracies = []
        for batch in batches:
            data, labels = batch
            accuracies.append(
                self.accuracy.eval(feed_dict={
                    self.model_input: data,
                    self.target: labels,
                    self.keep_prob: 1.0
                    }, session=self.sess
                )
            )
        return np.array(accuracies).mean()

    def predict(self, data=[]):
        output = self.model_output.eval(
            feed_dict={
                self.model_input: data,
                self.keep_prob: 0.5
            }, session=self.sess
        )
        return output