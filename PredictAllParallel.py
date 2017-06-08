import tensorflow as tf
import pandas as pd
import numpy as np
from time import time
from threading import Thread
from keras.models import load_model
import glob
import pickle
import os
import math
from keras.datasets import cifar10
from sklearn.metrics import accuracy_score

CPU_PERCENTAGES = np.linspace(0, 1, 11)       # 10 values from 0 to 1 (inclusive)
BATCHES = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
NRUNS = 3


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_test = x_test.astype('float32')
x_test /= 255



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


def run_prediction(session, y, data, x):
	for batch in data:
		session.run(y, feed_dict={x : batch})


def load_model_cpu_gpu(modelname):
    with tf.device('/cpu:0'):
        x = tf.placeholder(name='x', dtype=tf.float32,  shape=(None, 32,32,3))

    with tf.device('/gpu:0'):
        model_gpu = load_model(modelname)
        gpu = model_gpu(x)
        
    with tf.device('/cpu:0'):
        model_cpu = load_model(modelname)
        cpu = model_cpu(x)
        
    return cpu, gpu, x
   
def predict_cpu_gpu(cpu, data_cpu, gpu, data_gpu, x):
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, intra_op_parallelism_threads=8)) as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()

        threads = []

        # comment out 0 or 1 of the following 2 lines:
        threads += [Thread(target=run_prediction, args=(sess, cpu, data_cpu, x))]
        threads += [Thread(target=run_prediction, args=(sess, gpu, data_gpu, x))]

        t0 = time()

        for t in threads:
            t.start()

        coord.join(threads)

        t1 = time()


    return (t1 - t0)

def predict_parallel(modelname, data, batch_sizes = [1], cpu_percentage = [0.5], nruns = 1):
    cpu, gpu, x = load_model_cpu_gpu(modelname)
    
    results = []
    for batch_size in batch_sizes:
        batches = split_in_batches(data, batch_size)
        nbatches = len(batches)
        
        for cpu_perc in cpu_percentage:
            batches_cpu = int(nbatches * cpu_perc)
            data_cpu = batches[:batches_cpu]
            data_gpu = batches[batches_cpu:]
            
            tmp_sum = 0
            for run in range(nruns):
                print ("Running run {0} using {1:.2f}% CPU".format(run, cpu_perc))
                
                time_spent = predict_cpu_gpu(cpu, data_cpu, gpu, data_gpu, x)
                print(time_spent)
                tmp_sum += time_spent
                
                results.append([batch_size, cpu_perc, run, time_spent])

            print ("Average time: {0:.4f}".format(tmp_sum / nruns))
        
    return results
            

def evaluate_model(modelname):
    results = predict_parallel(modelname, x_test, BATCHES, CPU_PERCENTAGES, NRUNS)
    
    results_df = pd.DataFrame(results, columns = ['BATCH_SIZE', 'CPU_PERC', 'RUN', 'TIME'])
    basename = modelname[:modelname.rfind('.')]
    results_df.to_csv(basename + '.csv', index=None)

evaluate_model('model1.h5')

import gc; gc.collect()