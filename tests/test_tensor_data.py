import pytest
from hypothesis import given
from hypothesis.strategies import DataObject, data

import tinytorch
from tinytorch import TensorData

from .tensor_strategies import indices, tensor_data

# ## Tasks 2.1

# Check basic properties of layout and strides.


@pytest.mark.task2_1
def test_layout() -> None:
    "Test basis properties of layout and strides"
    data = [0] * 3 * 5
    # data : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # data : [0, 0, 0, 0, 0
    #         0, 0, 0, 0, 0
    #         0, 0, 0, 0, 0] 
    '''
        shape: UserShape -> (3, 5)
        strides: (5, 1)
    '''
    tensor_data = tinytorch.TensorData(data, (3, 5), (5, 1))
    print(tensor_data._storage, len(tensor_data._storage))
    print(tensor_data._shape, len(tensor_data._shape), type(tensor_data._shape))
    assert tensor_data.is_contiguous()
    assert tensor_data.shape == (3, 5)
    # 相邻元素在内存中连续
    assert tensor_data.index((1, 0)) == 5
    assert tensor_data.index((1, 1)) == 6
    assert tensor_data.index((1, 2)) == 7

    test_dup = tensor_data.permute(1, 0)
    # print(test_dup.shape)
    assert not test_dup.is_contiguous()
    # 相邻元素在内存中不连续
    assert test_dup.index((1, 0)) == 1
    assert test_dup.index((1, 1)) == 6

    tensor_data = tinytorch.TensorData(data, (5, 3), (1, 5))
    assert tensor_data.shape == (5, 3)
    assert not tensor_data.is_contiguous()

    data = [0] * 4 * 2 * 2
    '''
        如果不提供Strides,默认是行优先存储
    '''
    tensor_data = tinytorch.TensorData(data, (4, 2, 2))
    '''
        4 x [[0, 1]
             [2, 3]]

        strides : (4, 2, 1) <- strides_from_shape(shape)
            第一维度的步幅为 4，表示在移动到下一个元素时
            需要跳过 4 个元素，即移动到同一层的下一个元素。
            
            第二维度的步幅为 2，表示在移动到下一个元素时
            需要跳过 2 个元素，即移动到同一行的下一个元素。
            
            第三维度的步幅为 1，表示在移动到下一个元素时
            只需要跳过 1 个元素，即移动到同一列的下一个元素。
        
        layout : 
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    '''
    assert tensor_data.strides == (4, 2, 1)


@pytest.mark.xfail
def test_layout_bad() -> None:
    "Test basis properties of layout and strides"
    data = [0] * 3 * 5
    tinytorch.TensorData(data, (3, 5), (6,))

# --------6.23--------
@pytest.mark.task2_1
@given(tensor_data())
def test_enumeration(tensor_data: TensorData) -> None:
    "Test enumeration of tensor_datas."
    indices = list(tensor_data.indices())

    # Check that enough positions are enumerated.
    assert len(indices) == tensor_data.size

    # Check that enough positions are enumerated only once.
    assert len(set(tensor_data.indices())) == len(indices)

    # Check that all indices are within the shape.
    for ind in tensor_data.indices():
        for i, p in enumerate(ind):
            assert p >= 0 and p < tensor_data.shape[i]


@pytest.mark.task2_1
@given(tensor_data())
def test_index(tensor_data: TensorData) -> None:
    "Test enumeration of tensor_data."
    # Check that all indices are within the size.
    for ind in tensor_data.indices():
        pos = tensor_data.index(ind)
        assert pos >= 0 and pos < tensor_data.size

    base = [0] * tensor_data.dims
    with pytest.raises(tinytorch.IndexingError):
        base[0] = -1
        tensor_data.index(tuple(base))

    if tensor_data.dims > 1:
        with pytest.raises(tinytorch.IndexingError):
            base = [0] * (tensor_data.dims - 1)
            tensor_data.index(tuple(base))


@pytest.mark.task2_1
@given(data())
def test_permute(data: DataObject) -> None:
    td = data.draw(tensor_data())
    ind = data.draw(indices(td))
    td_rev = td.permute(*list(reversed(range(td.dims))))
    assert td.index(ind) == td_rev.index(tuple(reversed(ind)))

    td2 = td_rev.permute(*list(reversed(range(td_rev.dims))))
    assert td.index(ind) == td2.index(ind)


# ## Tasks 2.2

# Check basic properties of broadcasting.


@pytest.mark.task2_2
def test_shape_broadcast() -> None:
    c = tinytorch.shape_broadcast((1,), (5, 5))
    """
        1.在左边添加 1 ，取 5
        2.取 5
        3. == (5, 5) 
    """
    # print(c)
    assert c == (5, 5)

    # 倍数关系 np貌似无法广播？
    # c = tinytorch.shape_broadcast((2, 4), (3, 4, 4))
    # assert c == (3, 4, 4)
    
    # 倍数关系 np貌似无法广播？
    with pytest.raises(tinytorch.IndexingError):
      c = tinytorch.shape_broadcast((2, 4), (3, 4, 4))
      print(c)

    c = tinytorch.shape_broadcast((5, 5), (1,))
    assert c == (5, 5)

    c = tinytorch.shape_broadcast((1, 5, 5), (5, 5))
    assert c == (1, 5, 5)

    c = tinytorch.shape_broadcast((5, 1, 5, 1), (1, 5, 1, 5))
    assert c == (5, 5, 5, 5)

    """
        在Python中，with块主要用于管理资源，确保在代码块执行完毕后，
        相关资源能够被正确地释放或者进行必要的清理操作。
        
        with 语句提供了一种优雅的方式来管理资源，确保资源的正确分配和释放，避免了常见的资源泄漏问题，
    """
    with pytest.raises(tinytorch.IndexingError):
        c = tinytorch.shape_broadcast((5, 7, 5, 1), (1, 5, 1, 5))
        print(c)

    with pytest.raises(tinytorch.IndexingError):
        c = tinytorch.shape_broadcast((5, 2), (5,))
        print(c)

    c = tinytorch.shape_broadcast((2, 5), (5,))
    assert c == (2, 5)


@given(tensor_data())
def test_string(tensor_data: TensorData) -> None:
    tensor_data.to_string()
