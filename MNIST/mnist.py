import tensorflow as tf
import numpy as np
from random import shuffle
import os

height = 28
width = 28
channels = 1
n_classes = 10
batch_size = 32
n_epochs = 4
rootDir = "dataset/train"
filenames = []
 
keep_rate = 0.4
keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, [None, height, width, channels])
y = tf.placeholder(tf.float32, [None, n_classes])

def readFiles(rootDir,size):
    #Armazena todos os nomes dos arquivos em uma lista
    for dirName, subdirList, fileList in os.walk(rootDir):
        #print('Current directory: %s' % dirName)
        for fname in fileList:
            filenames.append(dirName + "/" + fname)
        
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()
    key, image_file = reader.read(filename_queue)
    
    # Pega a classe que representa o label. O nome da pasta deve ser o nome da classe
    S = tf.string_split([key],'/')
    
    length = tf.cast(S.dense_shape[1],tf.int32)
    label = S.values[length-tf.constant(2,dtype=tf.int32)]
    label = tf.string_to_number(label,out_type=tf.int32)
    image = tf.image.decode_png(image_file, channels=channels)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, [height, width])


    # Usando batch para acelerar o processo de treinamento
    image_batch, label_batch = tf.train.batch([image,label], batch_size=size)

    # Transformando o label em um vetor one hot
    label_batch = tf.one_hot(label_batch,10)

    return image_batch, label_batch

def CNN(filter1, filter2, kernel_size1, kernel_size2):
    
    image_batch, label_batch = readFiles(rootDir,batch_size)
    image_test, label_test = readFiles("dataset/test",10)

    # Rede Convolucional
    conv1 = tf.layers.conv2d(inputs=x, filters=filter1,kernel_size=[kernel_size1,kernel_size1] ,strides=[1,1],padding="same",activation=tf.nn.relu, trainable=True)
    conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=[2,2], padding="same")
    conv2 = tf.layers.conv2d(inputs=conv1,filters=filter2,kernel_size=[kernel_size2,kernel_size2], strides=[1,1],padding="same",activation=tf.nn.relu, trainable=True)
    conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=[2,2], padding="same")
    conv2 = tf.reshape(conv2,[-1, 7*7*filter2])
    fc = tf.layers.dense(inputs=conv2, units=1024, activation=tf.nn.relu, trainable=True)
    dropout = tf.layers.dropout(inputs=fc, rate=keep_rate)
    fc = tf.layers.dense(inputs=fc, units=512, activation=tf.nn.relu, trainable=True)
    prediction = tf.layers.dense(inputs=fc, units=10, trainable=True)

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for epoch in range(n_epochs):
            epoch_loss = 0
            for i in range(int(700/batch_size)):
                im, la = sess.run([image_batch, label_batch])
                _, c = sess.run([optimizer, cost], feed_dict = {x: im, y: la})
                epoch_loss += c    
            #print('Epoch', epoch, 'completed out of',n_epochs,'loss:',epoch_loss)
        
        im_t, la_t = sess.run([image_test, label_test])
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        p = accuracy.eval(feed_dict = {x: im_t, y: la_t})
        coord.request_stop()
        coord.join(threads)
        return p

filter1 = 16
while(filter1 < 65):
    filter2 = 16
    while(filter2 < 65):
        for kernel_size1 in range(5,10,2):
            for kernel_size2 in range(5,10,2):
                acc = 0
                for i in range(2):
                    acc += CNN(filter1,filter2,kernel_size1,kernel_size2)
                print("Filter1: " + str(filter1) + " Filter2: " + str(filter2) + " Kernel_size1: [" + str(kernel_size1) + ", " + str(kernel_size1) + "] Kernel_size2: [" + str(kernel_size2) + ", " +  str(kernel_size2) + "]")
                print("   " + "Accuracy: " + str(acc/2))
        filter2 *= 2
    filter1 *= 2

