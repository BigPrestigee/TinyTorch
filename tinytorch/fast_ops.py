from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
# 使用 njit 装饰器，并强制内联
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            # print(a)
            # print("out_shape", out._tensor._shape)
            # print("out_storage", out._tensor._storage)
            # print("out_zeros", out)
            out._tensor._storage[:] = start

            # print("out", out)

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

        # Make these always be a 3 dimensional multiply
        # a.shape = (2, 3), b.shape = (3, 4)
        # print("a = ", a)   
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2
        # print("a =... ", a)
        # a.shape = (1, 2, 3), b.shape = (1, 3, 4)

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        # ls = [1, ]
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        # ls = [1, 2, 4]
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))
        # (1, 2, 4) = (1, 2, 3) * (1, 3, 4)
        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """
    """
        高层调用的时候已做修饰，无需再次修饰
        f = tensor_map(njit()(fn))
    """
    # fn = njit()(fn) 

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError('Need to implement for Task 3.1')
        '''
            在nopython模式下，Numba需要所有的类型和操作都明确且支持，因此需要用NumPy的具体数据类型
            如np.int32或np.int64 来替代Python内置类型。这样，Numba才能正确编译和优化代码。
        '''
        in_index = np.zeros(len(in_shape), dtype=np.int32)
        out_index = np.zeros(len(out_shape), dtype=np.int32)

        for i in prange(len(out)):
            to_index(i, out_shape, out_index) # find big_tensor index in out
            broadcast_index(out_index, out_shape, in_shape, in_index) # find small_tensor index corrs big_tensor index

            x = in_storage[index_to_position(in_index, in_strides)] # find value x
            out[i] = fn(x)

    return njit(parallel=True)(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError('Need to implement for Task 3.1')
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        a_index = np.zeros(len(a_shape), dtype=np.int32)
        b_index = np.zeros(len(b_shape), dtype=np.int32)
        
        for i in prange(len(out)):
            to_index(i, out_shape, out_index)
            
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            x1 = a_storage[index_to_position(a_index, a_strides)]
            x2 = b_storage[index_to_position(b_index, b_strides)]
            out[i] = fn(x1, x2)

    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables
    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`
    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError('Need to implement for Task 3.1')
        out_index = np.zeros(len(out_shape), dtype=np.int32)

        for i in prange(len(out)):
            to_index(i, out_shape, out_index)
            # print("out_index", out_index)
            for j in prange(a_shape[reduce_dim]):
                a_index = out_index.copy()
                a_index[reduce_dim] = j
                # print("a_index", a_index)
                out[i] = fn(a_storage[index_to_position(a_index, a_strides)], out[i])

    return njit(parallel=True)(_reduce)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    # (1, 2, 4) = (1, 2, 3) * (1, 3, 4)
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0 # -> 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0 # -> 0

    assert a_shape[-1] == b_shape[-2]

    # for n in prange(out_shape[0]):
    #     for i in range(out_shape[1]): # 0 -> 1
    #         for j in range(out_shape[2]): # 0 -> 3
    #             tmp_sum = 0.0
    #             for k in range(a_shape[2]): # 0 -> 2
    #                 tmp_sum += ( 
    #                     a_storage[n * a_batch_stride + i * a_strides[1] + k * a_strides[2]] *
    #                     b_storage[n * b_batch_stride + k * b_strides[1] + j * b_strides[2]]
    #                 )
    #             out[n * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = tmp_sum

    # TODO: Implement for Task 3.2.
    # raise NotImplementedError('Need to implement for Task 3.2')
    # version 2
    # (1, 2, 4) = (1, 2, 3) * (1, 3, 4)
    out_index = np.zeros(len(out_shape), dtype=np.int32)
    
    for i in prange(len(out)):
        to_index(i, out_shape, out_index)
        # 如何理解？这种相乘？
        temp = 0.0
        for position in range(a_shape[-1]):
            temp += (
                a_storage[out_index[0] * a_batch_stride + out_index[1] * a_strides[1] + position * a_strides[2]] *
                b_storage[out_index[0] * b_batch_stride + out_index[2] * b_strides[2] + position * b_strides[1]]
            )
        out[i] = temp
    
    '''
        注：第二种，并行化粒度更细，潜在的并行度更高，第一种仅在 batch 维度进行并行
    '''

tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
