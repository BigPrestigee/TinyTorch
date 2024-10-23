from typing import Sequence

from .module import Parameter
from .scalar import Scalar
from .tensor_functions import _sqrt

class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters


class SGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.parameters:
            if p.value is None: # 跳过这个参数，继续处理下一个
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    # None是一种占位符，意味着需要在后续步骤中重新计算和赋值。
                    # None可表示该属性目前没有值或没有梯度信息
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)


# adam optimizer 的实现
class Adam(Optimizer):
    def __init__(self, 
                 parameters: Sequence[Parameter], 
                 lr: float = 1e-3, 
                 beta1: float = 0.9,         # 一阶动量因子
                 beta2: float = 0.999,       # 二阶动量因子
                 eps: float = 1e-8,          # 防止分母为0的一个小数
                 weight_decay: float = 0.0   # 权重衰减（L2正则化）
                 ):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # 用于偏置校正
        '''
            Adam 优化器的动量更新是针对每一个参数独立进行的。每个参数都有自己独立的梯度、动量和更新规则。
            因此，需要为每个参数维护独立的 m 和 v 以便跟踪它们的历史梯度信息。
            
            m和v是字典，key是参数对象，值是对应的m,v
        '''
        self.m = {p: 0 for p in self.parameters}  # 一阶动量
        self.v = {p: 0 for p in self.parameters}  # 二阶动量

    def zero_grad(self) -> None:
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        self.t += 1
        for p in self.parameters:
            # print(type(self.m[p]))
            '''
                这个m[p] is tensor 初始化全0
            '''
            if p.value is None:
                continue
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    grad = p.value.grad

                    # 权重衰减
                    if self.weight_decay != 0:
                        grad += self.weight_decay * p.value.data

                    # 更新一阶与二阶动量
                    self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * grad
                    self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (grad * grad)

                    # 计算偏置校正
                    m_hat = self.m[p] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[p] / (1 - self.beta2 ** self.t)

                    # 更新参数
                    # 重载一下 ** 运算符
                    p.update(p.value - self.lr * m_hat / (_sqrt(v_hat) + self.eps))