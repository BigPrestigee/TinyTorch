from typing import Callable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import lists

from tinytorch import MathTest
from tinytorch.operators import (
    add,
    addLists,
    eq,
    id,
    inv,
    inv_back,
    log_back,
    lt,
    max,
    mul,
    neg,
    negList,
    prod,
    relu,
    relu_back,
    sigmoid,
    sum,
)

from .strategies import assert_close, small_floats

# ## Task 0.1 Basic hypothesis tests.


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    "Check that the main operators all return the same value of the python version"
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if abs(x) > 1e-5:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a: float) -> None:
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a: float, b: float) -> None:
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a: float) -> None:
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a: float) -> None:
    "Check that a - 1.0 is always less than a"
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a: float) -> None:
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a: float) -> None:
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


# ## Task 0.2 - Property Testing

# Implement the following property checks
# that ensure that your operators obey basic
# mathematical rules.


@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a: float) -> None:
    """Check properties of the sigmoid function, specifically
    * It is always between 0.0 and 1.0.
    * one minus sigmoid is the same as sigmoid of the negative
    * It crosses 0 at 0.5
    * It is  strictly increasing.
    """
    # TODO: Implement for Task 0.2.
    # raise NotImplementedError('Need to implement for Task 0.2')

    assert 0.0 <= sigmoid(a) <= 1.0
    
    assert_close(1 - sigmoid(a), sigmoid(-a))
    
    if a == 0.0:
        assert_close(sigmoid(a), 0.5)
    
    # for increase
    """
        why need >=
        数值稳定性: 当输入的 a 非常大时，sigmoid 函数趋近于1，并且由于浮点数精度限制，
        即使我们对 a 进行微小的修改，得到的结果也可能在计算机的表示范围内没有区别，导致看似递增的函数在测试中失败。

        数值饱和: Sigmoid 函数在输入非常大或非常小的数时会出现饱和现象，即其值会非常接近1或0，
        并且再往更极端的方向改变输入时，输出不会有太大变化。
    """
    if a > 0.0:
        assert sigmoid(a) >= sigmoid(a - 0.0001)
    elif a < 0.0:
        assert sigmoid(a) <= sigmoid(a + 0.0001)

@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    "Test the transitive property of less-than (a < b and b < c implies a < c)"
    # TODO: Implement for Task 0.2.
    # raise NotImplementedError('Need to implement for Task 0.2')
    if a < b and b < c:
        assert a < c    

@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(x: float, y: float) -> None:
    """
    Write a test that ensures that :func:`tinytorch.operators.mul` is symmetric, i.e.
    gives the same value regardless of the order of its input.
    """
    # TODO: Implement for Task 0.2.
    # raise NotImplementedError('Need to implement for Task 0.2')
    assert mul(x, y) == mul(y, x)

@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(x: float, y: float, z: float) -> None:
    r"""
    Write a test that ensures that your operators distribute, i.e.
    :math:`z \times (x + y) = z \times x + z \times y`
    """
    # TODO: Implement for Task 0.2.
    #  raise NotImplementedError('Need to implement for Task 0.2')
    assert_close(mul(z, add(x, y)), mul(z, x) + mul(z, y))

@pytest.mark.task0_2
def test_other() -> None:
    """
    Write a test that ensures some other property holds for your functions.
    """
    # TODO: Implement for Task 0.2.
    # raise NotImplementedError('Need to implement for Task 0.2')
    # assert mul(z, add(x, y)) == mul(z, x) + mul(z, y)

# ## Task 0.3  - Higher-order functions

# These tests check that your higher-order functions obey basic
# properties.


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a: float, b: float, c: float, d: float) -> None:
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1: List[float], ls2: List[float]) -> None:
    """
    Write a test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    sum_ls1 = sum(ls1)
    sum_ls2 = sum(ls2)
    add_list = addLists(ls1, ls2)
    assert_close(sum_ls1 + sum_ls2, sum(add_list))

@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls: List[float]) -> None:
    assert_close(sum(ls), sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x: float, y: float, z: float) -> None:
    assert_close(prod([x, y, z]), x * y * z)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls: List[float]) -> None:
    check = negList(ls)
    for i, j in zip(ls, check):
        assert_close(i, -j)


# ## Generic mathematical tests

# For each unit this generic set of mathematical tests will run.


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn: Tuple[str, Callable[[float], float]], t1: float) -> None:
    name, base_fn = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
    fn: Tuple[str, Callable[[float, float], float]], t1: float, t2: float
) -> None:
    name, base_fn = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a: float, b: float) -> None:
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
