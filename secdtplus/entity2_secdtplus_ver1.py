
from entity2_secdt import ModelOwner, CloudServiceProvider, CloudServiceUser

from secure import param, prime, pseudo_random_generator, rand_num#, GF
import numpy as np
#import galois
import random
import sympy as sy


class ModelOwnerVer1(ModelOwner):
    def __init__(self):
        super().__init__()

    ## SecDT plus Ver.1
    def input_model_and_gen_plusV1_shares(self, root_node, attri_list):# -> list[list, list[list, list]]: 
        '''
        [+Ver.1]\n
        Fix a attribute sequence order vb and shuffle vb to vb_shuffle.\n
        Use vb_shuffle to build hinding(permutate) matrix A.\n
        Split A into shares and replace all attribute of nodes to the vb_shuffle order indice.
        '''
        #logging.debug("TEST LOGGING")
        self._root_node = root_node
        n = len(attri_list)

        ## Generate attribute permutation matrix A
        matrix_A = np.zeros((n,n), dtype=int)
        
        attri_idx_tuple_list = list(enumerate(attri_list))
        random.shuffle(attri_idx_tuple_list)
        indices, l = zip(*attri_idx_tuple_list)
        vb_shuffle_idx = indices

        for i in range(n):
            matrix_A[i][vb_shuffle_idx[i]] = 1
        #print("matrix_A: \n", matrix_A)

        self.vb = attri_list
        #print("[V1] vb: ", self.vb)

        ## Generate shares of matrix A
        ### Two ways: 1.All randomness. 2.Generate a seed and use this seed to fill all entries of matrix.
        ## All randomness
        self.matrix_A_share1 = np.random.randint(low=0, high=prime(), size=(n, n), dtype=int)
        self.matrix_A_share2 = (matrix_A - self.matrix_A_share1) % prime()
        #print("(self.matrix_A_share1 + self.matrix_A_share2) % prime(): \n", (self.matrix_A_share1 + self.matrix_A_share2) % prime())
        
        self._plus_transform_attri_to_index(l, attri_list)
        
        self.split_model_into_shares()

    def set_shares_to_two_parties_plusV1(self, csu, csp):
        '''[+Ver.1] Set M1, vb, A1 to CSU. Set M2, A to CSP.'''
        self.set_shares_to_two_parties(csu, csp)
        csu.set_plusV1_shares_to_two_parties(self.matrix_A_share1, self.vb)
        csp.set_plusV1_shares_to_two_parties(self.matrix_A_share2)
        return



class CloudServiceProviderVer1(CloudServiceProvider):
    def __init__(self):
        super().__init__()

    def set_plusV1_shares_to_two_parties(self, matrix_A_share2):
        '''[+Ver.1]'''
        self.matrix_A_share2 = matrix_A_share2



class CloudServiceUserVer1(CloudServiceUser):
    def __init__(self):
        super().__init__()

    ## SecDT plus Ver.1
    def set_plusV1_shares_to_two_parties(self, matrix_A_share1, vb):
        '''[+Ver.1]'''
        self.matrix_A_share1 = matrix_A_share1
        self.vb = vb

    def set_query_data_plusV1(self, qData):
        '''[+Ver.1]'''
        self.qData = qData
        #print("self.vb: ", self.vb)
        #print("qData: \n", qData)
        #print("qData Type: ", type(qData))
        qData_permutated = [qData[x] for x in self.vb]
        #print("qData_permutated: ", qData_permutated)
        self.qDataShare2_seed = rand_num()
        temp = self.qDataShare2_seed
        self.qDataShare2 = [temp]
        for _ in range(len(qData_permutated)):
            temp = pseudo_random_generator(temp)
            self.qDataShare2.append(temp)
        self.qDataShare1 = [(v - self.qDataShare2[idx]) % prime() for idx, v in list(enumerate(qData_permutated))]
        #print("self.qDataShare2: ", self.qDataShare2)
        #print("self.qDataShare1: ", self.qDataShare1)

    def send_query_data_to_csp_plusV1(self, csp):
        '''[+Ver.1]'''
        csp.set_query_data_share2(self.qDataShare2)