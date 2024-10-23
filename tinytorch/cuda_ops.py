from typing import Callable, Optional

import numba
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
# 标记为cuda设备函数，只能被cuda设备调用而不能被cpu调用
to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape) # already move to cuda
            # Block = 32 * thread
            threadsperblock = THREADS_PER_BLOCK
            # Grid = n * block
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(cuda.jit(device=True)(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        # a.shape = (2, 3), b.shape = (3, 2)
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2
        # a.shape = (1, 2, 3), b.shape = (1, 3, 2)
        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2])) # batch对齐
        # ls = [1,]
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        # ls = [1, 2, 2]
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls)) # already move to gpu

        # One block per batch, extra rows, extra col
        '''
            eg. (4, 100, 200) dispatch block，1 block -> 32 x 32 numbers, 1024 threads
        '''
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError('Need to implement for Task 3.3')
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos = index_to_position(in_index[: len(in_strides)], in_strides)
            x = in_storage[in_pos]
            out[i] = fn(x)

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError('Need to implement for Task 3.3')
        if i < out_size:
             to_index(i, out_shape, out_index)
             broadcast_index(out_index, out_shape, a_shape, a_index)
             broadcast_index(out_index, out_shape, b_shape, b_index)
             
             x1 = a_storage[index_to_position(a_index[: len(a_strides)], a_strides)]
             x2 = b_storage[index_to_position(b_index[: len(b_strides)], b_strides)]
             out[i] = fn(x1, x2)

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    # 每个thread block的共享内存
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x
    
    # TODO: Implement for Task 3.3.
    # raise NotImplementedError('Need to implement for Task 3.3')
    '''
        eg.
        线程块 0 的线程将 a[0] 到 a[31] 的值加载到其共享内存 cache 的位置 0 到 31
        线程块 1 的线程将 a[32] 到 a[63] 的值加载到其共享内存 cache 的位置 0 到 31
    '''
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0

    cuda.syncthreads()
    
    stride = 1
    while stride < BLOCK_DIM:
        if pos % (2 * stride) == 0:
            cache[pos] += cache[pos + stride]
        stride *= 2
        cuda.syncthreads()
    
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    print("size:", size)
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.

    Returns:
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        '''
            why use block cache?
            在使用 Numba 的 CUDA 库时，默认的数据传输是将数据从主机（CPU）内存复制到设备（GPU）内存。
            具体地，数据通常会被传输到 GPU 的全局内存（Global Memory）。
            
            对于需要频繁访问和更新相同数据的情况下，显式声明和使用共享内存可以显著提高性能。（提前进行数据专业）
        '''
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError('Need to implement for Task 3.3')
        out_dims = len(out_shape)
        a_dims = len(a_shape)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        pos = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if pos < out_size:
            to_index(pos, out_shape, out_index)
            reduce_num = a_shape[reduce_dim] # (3, 4, 1) -> (3, 1, 1) reduce_num = 4
            out[pos] = reduce_value
            for j in range(reduce_num):
                for k in range(out_dims):
                    a_index[k] = out_index[k]
                a_index[reduce_dim] = j
                pos_a = index_to_position(a_index[ : len(a_strides)], a_strides)
                out[pos] = fn(out[pos], a_storage[pos_a])

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square
    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    # raise NotImplementedError('Need to implement for Task 3.3')
    share_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    share_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    x = cuda.threadIdx.x
    y = cuda.threadIdx.y

    if x < size and y < size:
        share_a[y, x] = a[y * size + x]
        share_b[y, x] = b[y * size + x]
    else:
        share_a[y, x] = 0
        share_b[y, x] = 0

    cuda.syncthreads()
      
    if y < size and x < size:
        temp = 0
        for val in range(size):
            temp += share_a[y, val] * share_b[val, x]
        out[y * size + x] = temp

jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # TODO: Implement for Task 3.4.
    # raise NotImplementedError('Need to implement for Task 3.4')
    index_per_grid_x = cuda.blockDim.x * cuda.gridDim.x
    index_per_grid = cuda.blockDim.x * cuda.gridDim.x * cuda.blockDim.y * cuda.gridDim.y
    pos = index_per_grid * batch + index_per_grid_x * j + i

    if pos < out_size:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        to_index(pos, out_shape, out_index)
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)
        for j in range(a_shape[-1]):
            a_index[len(a_shape) - 1] = j
            pos_a = index_to_position(a_index[: len(a_strides)], a_strides)
            b_index[len(b_shape) - 2] = j
            pos_b = index_to_position(b_index[: len(b_strides)], b_strides)
            
            out[pos] += (a_storage[pos_a] * b_storage[pos_b])

tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
