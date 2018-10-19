"""
		Nome: Igor Martinelli      Zoltán Hirata Jetsmen
		NUSP: 9006336              9293272
		SCC0275 - Introdução à Redes Neurais
		2018/2
		Projeto 2: CNN
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random as rd

conv_units1 = 64
conv_units2 = 128
dense_units1 = 512
dense_units2 = 1024
epochs = 20

#Show log's on windows
tf.logging.set_verbosity(tf.logging.INFO)

#Model for cnn
def architecture(features, labels, mode):
    # Convolutional Layer 1 
    conv1 = tf.layers.conv2d(inputs=features, filters=conv_units1, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # Pooling Layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Normalizing Layer 1
    norm1 = tf.nn.local_response_normalization(pool1, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None)

    # Convolutional Layer 2
    conv2 = tf.layers.conv2d(inputs=norm1, filters=conv_units2, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # Normalizing Layer 2
    norm2 = tf.nn.local_response_normalization(conv2, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None)

    # Pooling Layer 2
    pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    pool2_flat = tf.reshape(pool2, [-1, 8*8*conv_units2])

    # Dense Layer 1
    dense1 = tf.layers.dense(inputs=pool2_flat, units=dense_units1, activation=tf.nn.relu)

    # Dropout Layer 
    dropout = tf.layers.dropout(inputs=dense1, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense Layer 2
    dense2 = tf.layers.dense(inputs=dropout, units=dense_units2, activation=tf.nn.relu)

    # Logits layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training and eval data
    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.cifar10.load_data()

    #Change dtype
    train_data = np.float32(train_data)
    test_data = np.float32(test_data)
    train_label = np.int32(train_label)
    test_label = np.int32(test_label)

    # Creating classifier and save on the 'model' folder.
    classifier = tf.estimator.Estimator(model_fn=architecture, model_dir='model')

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=train_data, y=train_label, num_epochs=epochs, shuffle=True)
    classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x= test_data, y=test_label, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()