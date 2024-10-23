from .tensor import rand
from .tensor_functions import MatMul
from .fast_conv import conv2d
from .module import Module, Parameter


class tLinear(Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = Parameter(rand((in_size, out_size)))
        self.bias = Parameter(rand((out_size,)))
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            self.weights.value.view(1, in_size, self.out_size)
            * x.view(batch, in_size, 1)
        ).sum(1).view(batch, self.out_size) + self.bias.value.view(1, self.out_size)


class tLinear2(Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = Parameter(rand((in_size, out_size)))
        self.bias = Parameter(rand((out_size,)))
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        '''
            这种实现 (tLinear2) 通常效率更高，尤其是在处理大规模矩阵运算时，得益于高度优化的矩阵乘法操作。
            大多数深度学习框架，包括 PyTorch，都会使用类似于第二种实现的方式来计算线性层。
        '''
        return MatMul(x, self.weights.value) + self.bias.value.view(1, self.out_size)
        # return x @ self.weights.value + self.bias.value.view(1, self.out_size)


class Dropout(Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        return (rand(x.shape) / 2 + 0.5 < self.rate) * x


# add Residual block   
class Residual(Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return x + self.layer.forward(x)

