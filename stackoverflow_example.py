import tensorflow as tf
import numpy as np
from time import time
from threading import Thread
from keras.models import load_model
import glob
import pickle
import os


n = 1024 * 8


def read_from_file(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

	
def read_train(path):
    train_data = []
    labels = []
    for file in glob.glob(os.path.join(path, 'data_batch*')):
        data = read_from_file(file)
        labels.append(data[b'labels'])
        train_data.append(data[b'data'])
    
    train_data = np.concatenate(train_data)
    labels = np.concatenate(labels)
    
    return train_data, labels

x_train, y_train = read_train('Cifar10')
y_labels = np.eye(10)[y_train]
x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

import math
def split_in_batches(data, batch_size):
    ret = []
    start = 0
    end = 0
    num_batches = math.ceil(data.shape[0] / batch_size)
    for batch_num in range(num_batches):
        end += batch_size
        if end > data.shape[0]:  #last batch
            ret.append(data[-batch_size:])
        else:
            ret.append(data[start:end])
        start = end
    return np.array(ret)

batches = split_in_batches(x_train, 1024)

data_cpu = batches[:10]
data_gpu = batches[:10]

with tf.device('/cpu:0'):
    x = tf.placeholder(name='x', dtype=tf.float32,  shape=(None, 32,32,3))


with tf.device('/gpu:0'):
    model_gpu = load_model('model1.h5')
    gpu = model_gpu(x)
    
with tf.device('/cpu:0'):
    model_cpu = load_model('model1.h5')
    cpu = model_cpu(x)	



def f(session, y, data):
	for batch in data:
		session.run(y, feed_dict={x : batch})


with tf.Session(config=tf.ConfigProto(log_device_placement=True, intra_op_parallelism_threads=8)) as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()

    threads = []

    # comment out 0 or 1 of the following 2 lines:
    #threads += [Thread(target=f, args=(sess, cpu, data_cpu))]
    threads += [Thread(target=f, args=(sess, gpu, data_gpu))]

    t0 = time()

    for t in threads:
        t.start()

    coord.join(threads)

    t1 = time()


print (t1 - t0)