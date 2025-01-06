
import time

import functools
import logging
import time

class Node():
    def __init__(self, attribute=None, threshold=None, left_child=None, right_child=None, is_leaf_node=False):
        #self._data = data
        self._attribute = attribute
        self._threshold = threshold
        self._left_child = left_child
        self._right_child = right_child
        self._is_leaf_node = is_leaf_node

    def threshold(self):
        return self._threshold

    def attribute(self):
        return self._attribute

    def is_leaf_node(self):
        return self._is_leaf_node

    def left_child(self):
        return self._left_child

    def right_child(self):
        return self._right_child
    
    def set_attribute(self, attri):
        self._attribute = attri

class Timer():
    def __init__(self, detail=None):
        self.start_time = time.time()
        self.detail = detail

    def reset(self, detail=None):
        self.start_time = time.time()
        self.detail = detail

    def end(self, detail=None):
        if detail is not None:
            self.detail = detail
        interval = (time.time() - self.start_time) * 1000
        #print(f"{self.detail}共花費 {interval:.4f}豪秒")
        #print(f"{self.detail}共花費 {interval:.20f}")
        return interval


def timed(logger, level=None, format='%s: %s ms'):
    if level is None:
        level = logging.DEBUG

    def decorator(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            start = time.time()
            result = fn(*args, **kwargs)
            duration = time.time() - start
            logger.log(level, format, repr(fn), duration * 1000)
            logger.log(level, args)
            return result
        return inner

    return decorator