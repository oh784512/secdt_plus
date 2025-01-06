import pickle
import sys
#import time
from structure2 import Node
#from secure import param, prime, pseudo_random_generator, rand_num

NODES = 0
DEPTH = 0


def size(obj):## unit is Bytes
    temp_obj = pickle.dumps(obj)
    return sys.getsizeof(temp_obj)


def tree_info(root):
    global NODES
    NODES = 0
    dfs(root, 0)
    return NODES, DEPTH


def dfs(current, depth):
    global NODES, DEPTH
    NODES += 1
    if current.is_leaf_node():
        if DEPTH < depth:
            DEPTH = depth
    else:
        dfs(current.left_child(), depth + 1)
        dfs(current.right_child(), depth + 1)