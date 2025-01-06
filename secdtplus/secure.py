"""
1. Generate secure parameter, prime, randint
2. Perform PseudoRandom Generator
"""

import os
import random
from Crypto.Util.number import getPrime

SECUREPARAM = 8#18
PRIME = getPrime(SECUREPARAM+1, os.urandom)

def pseudo_random_generator(seed):
    random.seed(seed)
    return random.getrandbits(SECUREPARAM) % PRIME


def rand_num():
    return random.randint(1, PRIME)


def prime():
    return PRIME


def param():
    return SECUREPARAM
