

import tensorflow as tf


class NeuralNetClassifier(tf.keras.Model):
    def __init__(self, student_version=True):
        super(NeuralNetClassifier, self).__init__()
        if student_version:
            self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        else:
            self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense2 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, inputs, training=False):
        h = self.dense1(inputs)
        if training:
            h = self.dropout(h)
        y = self.dense2(h)

        return y


