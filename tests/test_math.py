import pytest

import numpy as np

from vbnigmm.backend import (
    as_array,
    float32,
    float64,
    log_kv,
    log_dv_kv,
)


def isclose(x, y):
    return np.isclose(x, y, rtol=1e-10, atol=1e-1)


def allclose(x, y):
    return np.allclose(x, y, rtol=1e-10, atol=1e-1)


def test_log_kv():
    true_log_kv = np.loadtxt('tests/true_log_kv.csv', delimiter=',')
    for v, x, out in true_log_kv:
        ans = log_kv(v, x, float32, n=128, tol=1e-5)
        print(f'{v:12.6f}, {x:12.6f}, {out:12.6f}, {ans.numpy():12.6f}, {(out-ans)/out}')
        assert isclose(ans, out)
    #v, x, out = zip(*true_log_kv)
    #assert allclose(log_kv(v, x, float64), out)


def test_log_dv_kv():
    true_log_dv_kv = np.loadtxt('tests/true_log_dv_kv.csv', delimiter=',')
    for v, x, out in true_log_dv_kv:
        ans = log_dv_kv(v, x, float32, n=64, tol=1e-5)
        print(f'{v:12.6f}, {x:12.6f}, {out:12.6f}, {ans.numpy():12.6f}, {(out-ans)/out}')
        assert isclose(ans, out)
    #v, x, out = zip(*true_log_dv_kv)
    #assert allclose(log_dv_kv(v, x, float64), out)


if __name__ == '__main__':
    test_log_kv()
    test_log_dv_kv()
