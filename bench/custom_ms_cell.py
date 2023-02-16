from mindspore.common.initializer import initializer
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import ops
from mindspore.common.parameter import Parameter

class custom_Dense(nn.Cell):
    def __init__(self, in_channels, out_channels, weight_init='normal'):
        super().__init__()
        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels]), name="weight")
        self.matmul = P.MatMul()
        self.ein=ops.Einsum("...ij,kj->...ik")
    def construct(self, x):
        x = self.matmul(self.weight,x.T)
        x = x.T
        #x=self.ein((x,self.weight))
        return x
