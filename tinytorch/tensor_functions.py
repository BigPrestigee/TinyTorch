"""
Implementation of the autodifferentiation Functions for Tensor.
"""

from __future__ import annotations

import random
import math
from typing import TYPE_CHECKING

import numpy as np

import tinytorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = tinytorch.History(cls, ctx, vals)
        return tinytorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError('Need to implement for Task 2.3')
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: Implement for Task 2.4.
        # raise NotImplementedError('Need to implement for Task 2.4')
        a, b = ctx.saved_values
        return grad_output.f.mul_zip(b, grad_output), grad_output.f.mul_zip(a, grad_output)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError('Need to implement for Task 2.3')
        ctx.save_for_backward(t1)
        return t1.f.sigmoid_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4.
        # raise NotImplementedError('Need to implement for Task 2.4')
        (t1,) = ctx.saved_values
        return grad_output.f.add_zip(
            grad_output.f.mul_zip(grad_output.f.sigmoid_map(t1), grad_output),
            grad_output.f.neg_map(
                grad_output.f.mul_zip(grad_output.f.mul_zip(grad_output.f.sigmoid_map(t1), grad_output.f.sigmoid_map(t1)), 
                                      grad_output)    
            )
        )
        # return a.f.add_zip(
        #     a.f.mul_zip(grad_output, a.f.sigmoid_map(a)),
        #     a.f.neg_map(
        #         a.f.mul_zip(grad_output, a.f.mul_zip(a.f.sigmoid_map(a), a.f.sigmoid_map(a)))
        #     ),
        # )

class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError('Need to implement for Task 2.3')
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4.
        # raise NotImplementedError('Need to implement for Task 2.4')
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)

class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError('Need to implement for Task 2.3')
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4.
        # raise NotImplementedError('Need to implement for Task 2.4')
        """
            需要解包 tuple 成 tensor类型
        """
        (t1,) = ctx.saved_values
        # print("t1", t1)
        # print("t1_type()", type(t1[0]))
        # print("grad_output", grad_output)
        # print("grad_output_type", type(grad_output))
        return grad_output.f.log_back_zip(t1, grad_output)

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError('Need to implement for Task 2.3')
        ctx.save_for_backward(t1)
        return t1.f.exp_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4.
        # raise NotImplementedError('Need to implement for Task 2.4')
        (t1,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output.f.exp_map(t1), grad_output)

class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError('Need to implement for Task 2.3')
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: Implement for Task 2.4.
        # raise NotImplementedError('Need to implement for Task 2.4')
        return (
            grad_output.zeros(grad_output.shape),
            grad_output.zeros(grad_output.shape),
        )

class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError('Need to implement for Task 2.3')
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: Implement for Task 2.4.
        # raise NotImplementedError('Need to implement for Task 2.4')
        return (
                    grad_output.zeros(grad_output.shape),
                    grad_output.zeros(grad_output.shape),
                )

class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError('Need to implement for Task 2.3')
        return a.f.is_close_zip(a, b)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError('Need to implement for Task 2.3')
        ctx.save_for_backward(a, order)
        order_list = order._tensor._storage
        order_list_int = tuple(map(int, order_list))
        tens_store = a._tensor.permute(*order_list_int)
        return tinytorch.Tensor.make(
            tens_store._storage,
            tens_store.shape,
            tens_store.strides,
            backend=a.backend,
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        # TODO: Implement for Task 2.4.
        # raise NotImplementedError('Need to implement for Task 2.4')
        a, order = ctx.saved_values
        # 还需要返回一个 L 相对于 order的梯度？
        '''
            为了计算输入张量 a 的梯度，需要将 grad_output 按照 
            permute 操作的逆顺序进行排列，以便得到与 a 对应的梯度。
        '''
        return tinytorch.Tensor.make(
            grad_output._tensor._storage,
            a._tensor.shape,
            a._tensor.strides,
            backend=grad_output.backend,
        ), 0.0

class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return tinytorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (original,) = ctx.saved_values
        '''
            由于 view 操作只是改变张量的形状，grad_output 的内容与前向传播时的
            内容一致，因此只需将 grad_output 转换为原始形状来计算输入张量的梯度。
        '''
        return (
            tinytorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),  # 对t1的梯度
            grad_output.f.matrix_multiply(transpose(t1), grad_output),  # 对t2的梯度
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend

    Returns:
        new tensor
    """
    # print("---call tensor_function.zeros---")
    return tinytorch.Tensor.make(
        [0] * int(operators.prod(shape)), shape, backend=backend
    )

def _sqrt(tensor : Tensor) -> Tensor:
    sqrt_vals = [math.sqrt(val) for val in tensor._tensor._storage]
    return tinytorch.Tensor.make(sqrt_vals, tensor.shape, backend=tensor.backend)


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = tinytorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor
    """
    tensor = tinytorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """
    Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    # 计算输出 list 的 shape
    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []
    
    # 将嵌套 list 展平为一维列表
    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
