import socket
import time
import logging

from structure2 import Node, Timer, timed
from secure import param, prime, pseudo_random_generator, rand_num
#import numpy as np

#import random
#import sympy as sy
#from Crypto.Util.number import inverse

### [Reconstruction + attribute-hiding]
### SecDT plus ver1: generate vb + matrix A --> A dot vb = 
### SecDT plus ver2: generate vb + matrix A + invertible(nonsingilar) matrix S + S^(-1)

# logging.getLogger("graphviz").setLevel(logging.ERROR)
# logging.getLogger("matplotlib").setLevel(logging.ERROR)
# logging.getLogger("PIL").setLevel(logging.ERROR)
# logging.basicConfig(filename="secdtplus.log",
#                     filemode='w',#'a',
#                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                     datefmt='%H:%M:%S',
#                     level=logging.DEBUG)
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# logger.addHandler(handler)

class ModelOwner():

    def __init__(self):#(self, model_tree_root_node):
        self._root_node = None

    def input_model_and_split_into_shares(self, root_node): 
        self._root_node = root_node
        self.split_model_into_shares()
    
    def get_model(self):# -> Node:
        return self._root_node

    def split_model_into_shares(self):# -> list:
        self._seed = rand_num()
        if self._root_node == None:
            print("Please input model.")
            return None
        self._root_node_shares = self._build_shares(self._root_node, self._seed)
        #print("MO self._root_node_shares: ", self._root_node_shares)
        return self._root_node_shares

    # Combine copy1, cop2 into one function
    def _build_shares(self, original_node, seed):# -> list[Node, Node]:
        share1_left_child = None
        share1_right_child = None
        share2_left_child = None
        share2_right_child = None
        #thresholdShare1 = pseudo_random_generator(seed)
        #thresholdShare2 = original_node.threshold - thresholdShare1
        thresholdShare1 = pseudo_random_generator(seed)
        thresholdShare2 = (original_node.threshold() - thresholdShare1 ) % prime()
        if original_node.is_leaf_node():
            return [Node(attribute=original_node.attribute(), 
                        threshold=thresholdShare1, is_leaf_node=True), 
                    Node(attribute=original_node.attribute(), 
                        threshold=thresholdShare2, is_leaf_node=True)]
        else:
            thresholdShare1Binary = bin(thresholdShare1)[2:].zfill(param())
            first = int(thresholdShare1Binary[:param()//2], 2)
            second = int(thresholdShare1Binary[param()//2:], 2)
            share1_left_child, share2_left_child = self._build_shares(original_node.left_child(), first)
            share1_right_child, share2_right_child = self._build_shares(original_node.right_child(), second)

            return [Node(attribute=original_node.attribute(), 
                        threshold=thresholdShare1, 
                        left_child=share1_left_child, 
                        right_child=share1_right_child, 
                        is_leaf_node=False),
                    Node(attribute=original_node.attribute(), 
                        threshold=thresholdShare2, 
                        left_child=share2_left_child, 
                        right_child=share2_right_child, 
                        is_leaf_node=False)]
    
    def set_shares_to_two_parties(self, csu, csp):
        csu.set_seed(self._seed)
        csp.set_model_share2_root_node(self._root_node_shares[1])
        csu.generate_root()

    def _plus_transform_attri_to_index(self, shuffled_attri_list, origin_attri_list):
        '''[+] Transform all tree node attribute to attri_idx_tuple_list index'''
        tuples = list(enumerate(origin_attri_list))
        #print("origin_attri_list tuple: ", tuples)
        attri_old_order_mapping = dict((y, x) for x, y in tuples)
        #print("shuffled_attri_list: ", shuffled_attri_list)
        #print("attri_old_order_mapping: ", attri_old_order_mapping)
        shuffled_order = [attri_old_order_mapping[x] for x in shuffled_attri_list]
        #print("shuffled_order: ", shuffled_order)
        tuples = list(enumerate(shuffled_order))
        self.attri_new_order_mapping = dict((y, x) for x, y in tuples)
        #print("attri_new_order_mapping: ", self.attri_new_order_mapping)

        self._attri_to_index_recursively(self._root_node)
        
    def _attri_to_index_recursively(self, current):
        '''[+]'''
        if current.is_leaf_node() == False:
            #print("origin, trans:", current.attribute(), self.attri_new_order_mapping[current.attribute()])
            current.set_attribute(self.attri_new_order_mapping[current.attribute()])
            self._attri_to_index_recursively(current.left_child())
            self._attri_to_index_recursively(current.right_child())


class CloudServiceProvider():

    def __init__(self):
        self.model_share2_root = None

    def set_model_share2_root_node(self, share2):
        self.model_share2_root = share2
    
    def set_query_data_share2(self, qDataShare2):
        self.qDataShare2 = qDataShare2


class CloudServiceUser():
    
    def __init__(self):
        self.seed = None

        #self.ttt = Timer()
    
    def set_seed(self, seed):
        self.seed = seed

    def generate_root(self):
        #print("CSU H generate_root")
        self.root_node = None
        if self.seed == None:
            print("Please set seed.")
            return None
        self.root_node = Node(threshold=pseudo_random_generator(self.seed))
        #print("CSU H self.root_node attri: ", self.root_node.attribute())
        #print("CSU H self.root_node threshold: ", self.root_node.threshold())
        
    def set_query_data(self, data):
        self.qData = data
        self.qDataShare1 = {key: rand_num() for key in self.qData.keys()}
        self.qDataShare2 = {key: (self.qData[key]-self.qDataShare1[key]) % prime() for key in self.qData.keys()}

    def send_query_data_to_csp(self, csp):
        csp.set_query_data_share2(self.qDataShare2)
