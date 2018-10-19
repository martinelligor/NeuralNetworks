from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cifar_model import architecture
import matplotlib.image as mplimg

import numpy as np
import tensorflow as tf
import random as rd

def main(unused_argv):
    dataset = []
    # data
    data = ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # read dataset.
    for i in range(len(data)):
        dataset.append(mplimg.imread('dataset/' + data[i] + '.jpg'))

    # read the trained model.
    classifier = tf.estimator.Estimator(model_fn=architecture, model_dir='model', warm_start_from='model')
    # predict dataset classes.
    predict = tf.estimator.inputs.numpy_input_fn(x=np.array(dataset, dtype=np.float32), num_epochs=20, shuffle=False)
    res = list(classifier.predict(input_fn=predict))

    # accuracy list
    acc = []
    for i in range(len(data)):
        print('Original class: ', i)
        print('Predict class: ', res[i]['classes'])
        # probability of the selected class.
        print('Probability of the class: ' + str(res[i]['probabilities'][i]) + '\n')
        # computing the accuracy for the example i
        if(res[i]['classes'] == i):
            acc.append(1)
    # total accuracy of the model aplied on the dataset. 
    print('Model accuracy: ', (np.sum(acc)/len(data)))

if __name__ == "__main__":
    tf.app.run()