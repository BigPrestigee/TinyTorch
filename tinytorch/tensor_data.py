from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64] # 多维数组, dtype = float64
# eg. Storage = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """

    # TODO: Implement for Task 2.1.
    # raise NotImplementedError('Need to implement for Task 2.1')
    if len(index) != len(strides):
        raise ValueError("no match with inden_len and strides_len")
    # assert len(index) == len(strides)
    '''
        numba jit cannot use yeild
    '''
    sum = 0
    # return sum(index[i] * strides[i] for i in range(len(index)))      
    for i in range(len(index)):
        sum += index[i] * strides[i]

    return sum

def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    此函数将一个序数转换为张量中的索引。具体来说，
    它会根据张量的形状（shape）将一个线性位置（ordinal）映射到相应的多维索引（out_index）。
    
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # TODO: Implement for Task 2.1.
    # raise NotImplementedError('Need to implement for Task 2.1')
    '''
        shape : (3, 2, 5)
    '''
    size = 1
    for dim in shape:
        size *= dim # 3 * 2 * 5

    if ordinal < 0 or ordinal >= size:
        raise ValueError("no match")
    
    # shape : (3, 2, 5) -> range(2, 1, 0, -1) (-1取不到）
    '''
        eg:
            ordial = 9
            shape = (3, 4)
            out_index = [0, 0]
            out_index[1] = 9 % 4 = 1
            9 // 4 = 2
            out_index[0] = 2 % 3 = 2
            2 // 3 == 0
        out_index = (2, 1)
            x  x  x x
            x  x  x x
            x [x] x x 
        strides : (4, 1) default
    '''
    for i in range(len(shape) - 1, -1, -1):
        out_index[i] = ordinal % shape[i]
        ordinal //= shape[i]
    
def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.
    目的：将一个大的张量索引转换成小的张量索引，遵循广播规则。
    广播规则允许张量具有不同的形状但仍然能够执行逐元素操作。

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None
    """
    # TODO: Implement for Task 2.2.
    # raise NotImplementedError('Need to implement for Task 2.2')
    """
        big_shape = (3, 2, 4)
        shape = (1, 4)
        # after broadcast -> (3, 2, 4)
        [big_index = (1, 0, 3)]
        [Out_index = (0, 0)]
        -> Out_index = (0, 3)

        eg.big_shape: [[[6 9 7 4]
                        [7 4 7 3]]

                       [[9 2 1 [..5..]]
                        [3 4 5 1]]

                       [[0 1 5 0]
                        [2 0 6 2]]]
            
        shape :        [[3 5 3 [..4..]]
                        [3 5 3 4]]
        shape after broadcast:
                        [[[3 5 3 4]
                        [3 5 3 4]]

                        [[3 5 3 4]
                        [3 5 3 4]]

                        [[3 5 3 4]
                        [3 5 3 4]]]
    """
    diff = len(big_shape) - len(shape) # 1
    
    for i in range(len(shape)):
        out_index[i] = big_index[diff + i] if shape[i] > 1 else 0


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    # TODO: Implement for Task 2.2.
    # raise NotImplementedError('Need to implement for Task 2.2')
    """
        broadcaseting rule:
        1.当一个张量的某个维度大小为1时，可以将这个维度视为复制了 n 次来匹配另一个张量中相同位置的维度大小为 n 的维度。
        2.可以在一个张量的形状左侧添加大小为1的额外维度，以使其与另一个张量的维度数量相同。
        3.任何额外的大小为1的维度只能在形状的左侧隐式添加。
    """
    tuple_res = ()

    if len(shape1) > len(shape2):
        lagre, small = shape1, shape2
    else:
        lagre, small = shape2, shape1

    diff = len(lagre) - len(small) # (1, 5, 5) (5, 5) diff == 1

    for i in range(len(lagre)):
        l = lagre[i]
        # 在左边添加 1
        s = 1 if i < diff else small[i - diff]

        # 确保 l > s
        l, s = (l, s) if l > s else (s, l)
        # print(s)

        # 要求是倍数就行，不要求 == 1 && np貌似必须要求 1 (此为np的判断逻辑)？
        if s < l and s != 1 or s > l and l != 1:
            raise IndexingError('Cannot broadcast shapes')
        else:
            tuple_res += (l, )
        
    return tuple_res


def strides_from_shape(shape: UserShape) -> UserStrides:
    '''
        shape : (4, 2, 2)

        s = 2
        layout = [1, 2]
        s = 2
        layout = [1, 2, 4]
        s = 4
        layout = [1, 2, 4, 16]

    '''
    layout = [1]
    offset = 1
    for s in reversed(shape):
        # print(f's = {s}')
        layout.append(s * offset)
        # print(f'layout = {layout}')
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    '''
        类属性, 但底下的init赋值其实是创建了一个实例变量。
        这个实例变量与类属性同名但不同，它是属于这个实例的独立变量。

        如果类属性仅声明了类型但没有初始化，它实际上不会创建任何属性。
        这只是一个类型注解，并不影响程序的运行行为。
        如果你尝试访问一个未初始化的类属性，会抛出 AttributeError。
        print(MyClass._shape) -> AttributeError
    '''
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def __repr__(self) -> str:
        return self.to_string()

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        
        检查相邻元素是否在内存中也是相邻存储的:
        data -> [0,  1,  2,  3,  4, 5, 6,  7,  8,  9, 10, 11, 12, 13, 14]
        对于 (3, 5) , (5, 1) 视角
        UserShape:
            [0,  1,  2,  3,  4
            (5), 6,  7,  8,  9
            10, 11, 12, 13, 14]
        相邻的元素在内存中是相邻存储的: 如 0 和 1

        连续是要按照行（列）优先存储，如果是行优先，
        -需要外层的strides严格大于内层的strides
        -即: strides 按照递减顺序排列
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        # print("index -> ", index_to_position(array(index), self._strides))
        return index_to_position(array(index), self._strides)

    # 挨个输出每个用户视角坐标
    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # TODO: Implement for Task 2.1.
        # raise NotImplementedError('Need to implement for Task 2.1')
        '''
            这样会导致内存不连续吧？
            (3, 5) , (5, 1)
            UserShape:
             [0,  1,  2,  3,  4
              (5),  6,  7,  8,  9
              10, 11, 12, 13, 14]
            相邻的元素在内存中是相邻存储的: 如 0 和 1
            
            访问：
                tensor_data.index((1, 0)) == 5
            
            order = (1, 0)
            
            new_shape : (5, 3) 
            new_strides : (1, 5)

            UserShape : 
             [0, 5, 10
              (1), 6, 11
              2, 7, 12
              3, 8, 13
              4, 9, 14]
            相邻的元素在内存中不是相邻存储的: 如 0 和 5

            总结: 
                重新排列维度并不会导致内存不连续，而是改变了数据在内存中的访问方式。
                对于操作后的对象，index访问会不同
            访问：
                test_dup.index((1, 0)) == 1
        '''
        order = tuple(int(i) for i in order)
        new_shape = [self.shape[i] for i in order]
        new_strides = [self.strides[i] for i in order]
        
        return TensorData(self._storage, tuple(new_shape), tuple(new_strides))

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
