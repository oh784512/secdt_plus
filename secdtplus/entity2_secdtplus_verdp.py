
from entity2_secdt import ModelOwner, CloudServiceProvider, CloudServiceUser

from structure2 import Node
from secure import param, prime, pseudo_random_generator, rand_num
#import numpy as np

#import random
#import sympy as sy
#from Crypto.Util.number import inverse
#from Crypto.Hash import SHA1

class ModelOwnerVerDP(ModelOwner):
    def __init__(self):
        super().__init__()

    def input_model_and_gen_plusVDP_shares(self, root_node, attri_list): 
        '''[+Ver.DP]'''
        ### Set hashed attribute to tree model attribute field
        self._root_node = root_node
        self._plus_transform_model_attri_to_attriTargettingVector(attri_list)
        self.split_model_into_shares_VDP()

    def _plus_transform_model_attri_to_attriTargettingVector(self, attri_list):
        self.attri_list = attri_list
        self._attri_to_attriTargettingVector_recursively(self._root_node)
    
    def _attri_to_attriTargettingVector_recursively(self, current):
        attriTargeting_v = [0]*len(self.attri_list)
        if current.is_leaf_node() == False:
            attriTargeting_v[current.attribute()] = 1
            current.set_attribute(attriTargeting_v)
            self._attri_to_attriTargettingVector_recursively(current.left_child())
            self._attri_to_attriTargettingVector_recursively(current.right_child())
        else:
            current.set_attribute([-1]*len(self.attri_list))

    def split_model_into_shares_VDP(self):
        self._seed = rand_num()
        self._seed_attri_v = [rand_num()]*len(self.attri_list)
        if self._root_node == None:
            print("Please input model.")
            return None
        self._root_node_shares = self._build_shares_VDP(self._root_node, self._seed, self._seed_attri_v)
        return self._root_node_shares

    def _build_shares_VDP(self, original_node, seed, seed_attri_v):# -> list[Node, Node]:
        share1_left_child = None
        share1_right_child = None
        share2_left_child = None
        share2_right_child = None
        thresholdShare1 = pseudo_random_generator(seed)
        thresholdShare2 = (original_node.threshold() - thresholdShare1 ) % prime()
        attributeShareV1 = [pseudo_random_generator(x) for x in seed_attri_v]
        #print("original_node.attribute(): ", original_node.attribute())
        attributeShareV2 = [(original_node.attribute()[i] - x) % prime() for i, x in enumerate(attributeShareV1)]
        if original_node.is_leaf_node():
            return [Node(attribute=attributeShareV1, 
                        threshold=thresholdShare1, is_leaf_node=True), 
                    Node(attribute=attributeShareV2, 
                        threshold=thresholdShare2, is_leaf_node=True)]
        else:
            thresholdShare1Binary = bin(thresholdShare1)[2:].zfill(param())
            first = int(thresholdShare1Binary[:param()//2], 2)
            second = int(thresholdShare1Binary[param()//2:], 2)

            attributeShareV1Binary = [bin(x)[2:].zfill(param()) for x in attributeShareV1]
            first_attri_v = [int(x[:param()//2], 2) for x in attributeShareV1Binary]
            second_attri_v = [int(x[param()//2:], 2) for x in attributeShareV1Binary]

            share1_left_child, share2_left_child = self._build_shares_VDP(original_node.left_child(), first, first_attri_v)
            share1_right_child, share2_right_child = self._build_shares_VDP(original_node.right_child(), second, second_attri_v)

            return [Node(attribute=attributeShareV1, 
                        threshold=thresholdShare1, 
                        left_child=share1_left_child, 
                        right_child=share1_right_child, 
                        is_leaf_node=False),
                    Node(attribute=attributeShareV2, 
                        threshold=thresholdShare2, 
                        left_child=share2_left_child, 
                        right_child=share2_right_child, 
                        is_leaf_node=False)]
        
    def set_shares_to_two_parties_plusVDP(self, csu, csp):
        '''[+Ver.DP]'''
        csu.set_seed_attri_v(self._seed_attri_v)
        self.set_shares_to_two_parties(csu, csp)
        return

class CloudServiceProviderVerDP(CloudServiceProvider):
    def __init__(self):
        super().__init__()

    def set_query_data_share2(self, qDataShare1):
        self.qDataShare1 = qDataShare1

    def setTQ(self, tq1):
        self.tq1 = tq1

class CloudServiceUserVerDP(CloudServiceUser):
    def __init__(self):
        super().__init__()

    # def set_plusVH_shares_to_two_parties(self, matrix_S, vb):
    #     '''[+Ver.H]'''

    def set_seed_attri_v(self, seed_attri_v):
        self.seed_attri_v = seed_attri_v

    def generate_root(self):
        #print("CSU H generate_root")
        self.root_node = None
        if self.seed == None:
            print("Please set seed.")
            return None
        self.root_node = Node(threshold=pseudo_random_generator(self.seed), 
                              attribute=[pseudo_random_generator(x) for x in self.seed_attri_v])
    
    def set_query_data_plusVDP(self, qData):
        '''[+Ver.DP]'''
        #super().set_query_data(qData)
        self.qData = qData
        self.qDataShare0 = [rand_num() for _ in range(len(qData))]
        self.qDataShare1 = [(x - y) % prime() for x, y in zip(qData, self.qDataShare0)]

    def send_query_data_to_csp_plusVDP(self, csp):
        '''[+Ver.DP]'''
        #super().send_query_data_to_csp(csp)
        csp.set_query_data_share2(self.qDataShare1)
    
    def setTQ(self, tq0):
        self.tq0 = tq0