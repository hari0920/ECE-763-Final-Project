import os
import sys
import timeit
import time
import numpy
from numpy.random import random


def test_eigenvalue():
    """
    Test eigen value computation of a matrix
    """


i = 500
data = random((i, i))
result = numpy.linalg.eig(data)


def test_svd():
    """
    Test single value decomposition of a matrix
    """


i = 500
data = random((i, i))
result = numpy.linalg.svd(data)
result = numpy.linalg.svd(data, full_matrices=False)


def test_inv():
    """
    Test matrix inversion
    """


i = 500
data = random((i, i))
result = numpy.linalg.inv(data)


def test_det():
    """
    Test the computation of the matrix determinant
    """


i = 500
data = random((i, i))
result = numpy.linalg.slogdet(data)[1]


def test_dot():
    """
    Test the dot product
    """


i = 500
a = random((i, i))
b = numpy.linalg.inv(a)
result = numpy.dot(a, b) - numpy.eye(i)

# Test to start. The dict is the value I had with the MKL using EPD 6.0 and without MKL using EPD 5.1
tests = {test_eigenvalue: (752., 3376.),
         test_svd: (4608., 15990.),
         test_inv: (418., 1457.),
         test_det: (186.0, 400.),
         test_dot: (666., 2444.)}

# Setting the following environment variable in the shell executing the script allows
# you limit the maximal number threads used for computation
#THREADS_LIMIT_ENV = 'OMP_NUM_THREADS'

def start_benchmark():

    #if THREADS_LIMIT_ENV in os.environ:
    #print("Maximum number of threads used for computation is : %s" % os.environ[THREADS_LIMIT_ENV])
    print("-" * 80)
    print(
    "Starting timing with numpy %s\nVersion: %s" % (numpy.__version__, sys.version))
    print(
    "%20s : %10s" % ("Function", "Timing [s]"))
    for fun, bench in tests.items():
        t = timeit.Timer(stmt="%s()" % fun.__name__, setup="from __main__ import %s" % fun.__name__)
        res = t.repeat(number=1)
        timing = sum(res) / len(res)
        print(
        "%20s : %7.9f" % (fun.__name__, timing))#, bench[0] / timing, bench[1] / timing))# , bench[0] / timing, bench[1] / timing))

if __name__ == '__main__':
    start_benchmark()