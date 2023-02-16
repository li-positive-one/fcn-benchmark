import tensorflow as tf
import sys
import os
import time
from omegaconf import OmegaConf

from tensorflow.python.client import device_lib

def get_net(neurons):
    model=tf.keras.models.Sequential([
        tf.keras.layers.Dense(neurons[i], activation='relu') for i in range(1, len(neurons)-1)])
    model.add(tf.keras.layers.Dense(neurons[-1]))
    model.build(input_shape=(None, neurons[0]))
    return model

def get_data(inputs, outputs, batchsize):
    return tf.random.normal([batchsize, inputs]), tf.random.normal([batchsize, outputs])

def prepare(config):
    net=get_net([config.indim,]+config.neurons+[config.outdim,])
    inputs, outputs=get_data(config.indim, config.outdim, config.batchsize)
    net.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.mean_squared_error)
    return net, inputs, outputs

def train(*args):
    net, inputs, outputs = args
    net.fit(inputs, outputs, epochs=config.epochs, batch_size=config.batchsize,verbose=0)
    return 

def bench(config):
    if config.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    start_time=time.time()
    env=prepare(config)
    t1_time=time.time()
    train(*env)
    t2_time=time.time()
    print("Prepare time: ", t1_time-start_time)
    print("Train time: ", t2_time-t1_time)
    
if __name__=="__main__":
    config=OmegaConf.load(sys.argv[1])
    bench(config)