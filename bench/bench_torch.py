import torch 
import torch.nn as nn
import sys
import time
from omegaconf import OmegaConf
import os

class torch_fcn(nn.Module):
    def __init__(self, neurons, act=nn.ReLU()):
        super(torch_fcn, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(neurons)-1):
            self.layers.append(nn.Linear(neurons[i], neurons[i+1]))
            if i < len(neurons)-2:
                self.layers.append(act)
    def forward(self, x):
        for layer in self.layers:
            x=layer(x)
        return x

def get_net(neurons):
    return torch_fcn(neurons)

def get_data(inputs, outputs, batchsize, device):
    return torch.rand(batchsize, inputs,device=device), torch.rand(batchsize, outputs, device=device)

def prepare(config):
    net=get_net([config.indim,]+config.neurons+[config.outdim,]).to(device="cuda" if config.use_gpu else "cpu")
    inputs, outputs=get_data(config.indim, config.outdim, config.batchsize, device="cuda" if config.use_gpu else "cpu")
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    return net, inputs, outputs, optimizer

def train(*args):
    net, inputs, outputs, optimizer = args
    criterion = nn.MSELoss()
    for i in range(config.epochs):
        optimizer.zero_grad()
        outputs_pred = net(inputs)
        loss = criterion(outputs_pred, outputs)
        loss.backward()
        optimizer.step()
    return 

def bench(config):
    if config.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device="cuda"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device="cpu"
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