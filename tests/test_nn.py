import tinytorch.tensor
import pytest
from hypothesis import given

import tinytorch
from tinytorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    '''
        print('-------------------------------')
        print("t -> ", t)
        print(" t after tile ->", tinytorch.tile(t, (2, 2))[0])
        after_pool = tinytorch.tile(t, (2, 2))[0].mean(4)
        print("this -> ", after_pool)
        print("af_shape -> ", after_pool.shape)
        after_view = after_pool.view(1, 1, 2, 2)
        # 显示应该-需要对-输出进行-transpose()-显示，否则是经过池化之后图像是反的？
        after_view_transpose = after_view.transpose()
        print("reshape -> ", after_view_transpose)
        print("after view -> ", after_view_transpose.shape)
        print("right top -> ", sum([t[0, 0, i, j] for i in range(2) for j in range(2, 4)]) / 4.0)
        print("right top small -> ", after_view[0, 0, 0, 1])
    '''
    out = tinytorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = tinytorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = tinytorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    tinytorch.grad_check(lambda t: tinytorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError('Need to implement for Task 4.4')
    print("t -> ", t)
    out = tinytorch.argmax(t, 2)
    print("out -> ", out)
    # tinytorch.grad_check(tinytorch.Max.apply, t, tinytorch.tensor([0]))


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = tinytorch.maxpool2d(t, (2, 2))
    # print("out -> ", out)
    # print("t -> ", t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = tinytorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = tinytorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = tinytorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = tinytorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = tinytorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = tinytorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = tinytorch.softmax(t, 2)
    x = q.sum(dim=2)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = tinytorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    tinytorch.grad_check(lambda a: tinytorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = tinytorch.softmax(t, 3)
    q2 = tinytorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    tinytorch.grad_check(lambda a: tinytorch.logsoftmax(a, dim=2), t)
