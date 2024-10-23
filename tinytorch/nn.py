from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError('Need to implement for Task 4.3')
    new_height = height // kh
    new_width = width // kw
    input = (
        input.contiguous()
        .view(batch, channel, height, new_width, kw)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch, channel, new_width, new_height, kh * kw)
    )

    return (input, new_height, new_width)


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError('Need to implement for Task 4.3')
    # mean = 4 , 第五个维度求均值
    input = tile(input, kernel)[0].mean(4)
    input = input.view(batch, channel, height // kh, width // kw)
    return input

max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


mul_tensor = FastOps.zip(operators.mul)

class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        # TODO: Implement for Task 4.4.
        # raise NotImplementedError('Need to implement for Task 4.4')
        dim_ = int(dim._tensor._storage[0])
        ctx.save_for_backward(input, dim_)
        out = max_reduce(input, dim_)
        return out 

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        # TODO: Implement for Task 4.4.
        # raise NotImplementedError('Need to implement for Task 4.4')
        input, dim = ctx.saved_values
        res = argmax(input, dim)
        return grad_output.f.mul_zip(grad_output, res) , 0.0
    
# Max_ = Max.apply

def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError('Need to implement for Task 4.4')
    input = input.exp()
    t = input.sum(dim)
    return input / t
    

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError('Need to implement for Task 4.4')
    t = input.exp().sum(dim).log()
    return input - t


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError('Need to implement for Task 4.4')
    input = tile(input, kernel)[0]
    # tensor_max = max_reduce(input, -1)
    tensor_max = max(input, -1)
    out = tensor_max.view(batch, channel, height // kh, width // kw)

    return out


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError('Need to implement for Task 4.4')
    if not ignore:
        rand_tansor = rand(input.shape)
        '''
            rate == 0 -> return all 1
            rate == 1 -> return all 0
        '''
        rand_drop = rand_tansor > rate 
        return input * rand_drop
    return input # do not dropout


