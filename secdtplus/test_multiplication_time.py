import time
import sympy as sy
import numpy as np
import random

from structure2 import Timer

timer = Timer()

for _ in range(1000):
    A = np.array([random.randint(0, 100) for _ in range(50)])
    B = np.random.randint(100, size=(50, 50))
    #print(A)
    #print(B)

    timer.reset("Evaluation once")
    result = A @ B
    timestamp4 = (timer.end())

    print(timestamp4)
    #print(result)