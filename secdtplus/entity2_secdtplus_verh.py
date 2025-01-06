
from entity2_secdt import ModelOwner, CloudServiceProvider, CloudServiceUser

from structure2 import Node
from secure import param, prime, pseudo_random_generator, rand_num
#import numpy as np

#import random
#import sympy as sy
#from Crypto.Util.number import inverse
from Crypto.Hash import SHA1

class ModelOwnerVerH(ModelOwner):
    def __init__(self):
        super().__init__()

    def input_model_and_gen_plusVH_shares(self, root_node, attri_list): 
        '''[+Ver.H]'''
        ### Set hashed attribute to tree model attribute field
        self._root_node = root_node
        self.attri_list_mapping_idx = dict(enumerate(attri_list))
        #print("self.attri_list_mapping_idx: ", self.attri_list_mapping_idx)
        self.__plus_transform_model_attri_to_hash(attri_list)
        self.split_model_into_shares_VH()

    def __plus_transform_model_attri_to_hash(self, attri_list):
        attri_list_hashed = []
        hashed = SHA1.new(b'attri')
        #print("hash attri: ", hashed.digest())
        #print("size hash attri: ", hashed.digest())
        for attri in attri_list:
            #print("hashed.update(attri.encode()): ", hashed.update(attri.encode()))
            #print("type: ", type(hashed.update(attri.encode())))
            #print("hash: ", hashed.digest())
            hashed.update(attri.encode())
            attri_list_hashed.append(int.from_bytes(hashed.digest()) % prime())
        #print("MO attri_list_hashed: ", attri_list_hashed)
        self.attri_hash_mapping =  dict((attri_list[i], attri_list_hashed[i]) for i in range(len(attri_list)))
        #print("attri_hash_mapping: ", self.attri_hash_mapping)
        self._attri_to_hash_recursively(self._root_node)
    
    def _attri_to_hash_recursively(self, current):
        if current.is_leaf_node() == False:
            #print("origin, trans:", current.attribute(), self.attri_new_order_mapping[current.attribute()])
            #print("current.attribute(): ", current.attribute())
            #print("self.attri_list_mapping_idx[current.attribute()]: ", self.attri_list_mapping_idx[current.attribute()])
            #print("self.attri_hash_mapping[self.attri_list_mapping_idx[current.attribute()]]: ", self.attri_hash_mapping[self.attri_list_mapping_idx[current.attribute()]])
            current.set_attribute(self.attri_hash_mapping[self.attri_list_mapping_idx[current.attribute()]])
            self._attri_to_hash_recursively(current.left_child())
            self._attri_to_hash_recursively(current.right_child())

    def split_model_into_shares_VH(self):# -> list:
        self._seed = rand_num()
        self._seed_attri = rand_num()
        if self._root_node == None:
            print("Please input model.")
            return None
        self._root_node_shares = self._build_shares_VH(self._root_node, self._seed, self._seed_attri)
        return self._root_node_shares

    def _build_shares_VH(self, original_node, seed, seed_attri):# -> list[Node, Node]:
        share1_left_child = None
        share1_right_child = None
        share2_left_child = None
        share2_right_child = None
        thresholdShare1 = pseudo_random_generator(seed)
        thresholdShare2 = (original_node.threshold() - thresholdShare1 ) % prime()
        attributeShare1 = pseudo_random_generator(seed_attri)
        attributeShare2 = (original_node.attribute() - attributeShare1 ) % prime()
        if original_node.is_leaf_node():
            return [Node(attribute=attributeShare1, 
                        threshold=thresholdShare1, is_leaf_node=True), 
                    Node(attribute=attributeShare2, 
                        threshold=thresholdShare2, is_leaf_node=True)]
        else:
            thresholdShare1Binary = bin(thresholdShare1)[2:].zfill(param())
            first = int(thresholdShare1Binary[:param()//2], 2)
            second = int(thresholdShare1Binary[param()//2:], 2)

            attributeShare1Binary = bin(attributeShare1)[2:].zfill(param())
            first_attri = int(attributeShare1Binary[:param()//2], 2)
            second_attri = int(attributeShare1Binary[param()//2:], 2)

            share1_left_child, share2_left_child = self._build_shares_VH(original_node.left_child(), first, first_attri)
            share1_right_child, share2_right_child = self._build_shares_VH(original_node.right_child(), second, second_attri)

            return [Node(attribute=attributeShare1, 
                        threshold=thresholdShare1, 
                        left_child=share1_left_child, 
                        right_child=share1_right_child, 
                        is_leaf_node=False),
                    Node(attribute=attributeShare2, 
                        threshold=thresholdShare2, 
                        left_child=share2_left_child, 
                        right_child=share2_right_child, 
                        is_leaf_node=False)]
        
    def set_shares_to_two_parties_plusVH(self, csu, csp):
        '''[+Ver.H]'''
        csu.set_seed_attri(self._seed_attri)
        self.set_shares_to_two_parties(csu, csp)
        #csu.set_plusVH_shares_to_two_parties(self.sym_matrix_S, self.vb)
        #csp.set_plusV2_shares_to_two_parties(self.sym_matrix_K)
        return


class CloudServiceProviderVerH(CloudServiceProvider):
    def __init__(self):
        super().__init__()

    # def set_plusVH_shares_to_two_parties(self, matrix_K):
    #     '''[+Ver.H]'''
    #     self.matrix_K = matrix_K

    def set_query_data_plusVH(self, attri_field_from_tree):
        '''[+Ver.H]'''
        self.eta1 = attri_field_from_tree

    def setTQ(self, tq1):
        self.tq1 = tq1

class CloudServiceUserVerH(CloudServiceUser):
    def __init__(self):
        super().__init__()

    # def set_plusVH_shares_to_two_parties(self, matrix_S, vb):
    #     '''[+Ver.H]'''

    def set_seed_attri(self, seed_attri):
        self.seed_attri = seed_attri

    def generate_root(self):
        #print("CSU H generate_root")
        self.root_node = None
        if self.seed == None:
            print("Please set seed.")
            return None
        self.root_node = Node(threshold=pseudo_random_generator(self.seed), 
                              attribute=pseudo_random_generator(self.seed_attri))
    
    def set_query_data_plusVH(self, qData_attri_list_hashed_entry, attri_field_from_tree):
        '''[+Ver.H]'''
        #self.qData = qData

        self.l = 0
        self.h = qData_attri_list_hashed_entry
        self.eta0 = attri_field_from_tree

    def send_query_data_to_csp_plusVH(self, csp):
        '''[+Ver.H]'''
        #csp.set_query_data_plusVH()

    def setTQ(self, tq0):
        self.tq0 = tq0