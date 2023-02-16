import mindspore as ms
import mindspore.nn as nn
import sys
import time
from omegaconf import OmegaConf
import os

class ms_fcn(nn.Cell):
    def __init__(self, neurons, act=nn.ReLU()):
        super().__init__()
        self.layers = []
        for i in range(len(neurons)-1):
            self.layers.append(nn.Dense(neurons[i], neurons[i+1]))
            if i < len(neurons)-2:
                self.layers.append(act)
        self.seq=nn.SequentialCell(self.layers)
    def construct(self, x):
        x=self.seq(x)
        return x

def get_net(neurons):
    return ms_fcn(neurons)

def get_data(inputs, outputs, batchsize):
    return ms.numpy.rand(batchsize, inputs), ms.numpy.rand(batchsize, outputs)

def prepare(config):
    net=get_net([config.indim,]+config.neurons+[config.outdim,])
    inputs, outputs=get_data(config.indim, config.outdim, config.batchsize)
    optimizer = nn.Adam(params=net.trainable_params(), lr=config.lr)
    return net, inputs, outputs, optimizer

def train(*args):
    net, inputs, outputs, optimizer = args
    criterion = nn.MSELoss()

    def forward_fn(inputs, outputs):
        outputs_pred = net(inputs)
        loss = criterion(outputs_pred, outputs)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    @ms.jit
    def train_step(inputs, outputs):
        loss, grads = grad_fn(inputs, outputs)
        return loss

    for i in range(config.epochs):
        train_step(inputs=inputs, outputs=outputs)
    return 

def bench(config):
    if config.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device="GPU"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device="CPU"

    if config.use_graph:
        ms.set_context(mode=ms.GRAPH_MODE,device_target=device)
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE,device_target=device)

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