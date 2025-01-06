import socket
import time
import logging

from structure2 import Node, Timer, timed
from secure import param, prime, pseudo_random_generator, rand_num
import numpy as np

import random
import sympy as sy
from Crypto.Hash import SHA1

from entity2_secdt import ModelOwner, CloudServiceProvider, CloudServiceUser

logging.getLogger("graphviz").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.basicConfig(filename="secdtplus.log",
                    filemode='w',#'a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

class Protocol():

    def __init__(self):
        self.timer_og_1 = Timer()
        self.timer_og_2 = Timer()
        self.timer_v1_1 = Timer()
        self.timer_v1_2 = Timer()
        self.timer_v2_1 = Timer()
        self.timer_v2_2 = Timer()
        self.timer_vh_1 = Timer()
        self.timer_vh_2 = Timer()
        self.timer_vdp_1 = Timer()
        self.timer_vdp_2 = Timer()
        self.internal_test_timer = Timer()
        logger.log(level=logging.DEBUG, msg="prime(): "+str(prime()))

    def set_attri_list(self, attri_arr):
        self.attri_list = attri_arr

    def prepare(self, mo, csu, csp, root_node):
        mo.input_model_and_split_into_shares(root_node)
        mo.set_shares_to_two_parties(csu, csp)
        return
    
    ## original SecDT
    def classify(self, csu, csp, qData):
        self.timer_og_1.reset("Split query data")
        csu.set_query_data(qData)
        csu.send_query_data_to_csp(csp)
        timestamp_og_1 = self.timer_og_1.end()

        self.timer_og_2.reset("Evaluation once")
        u_node = csu.root_node
        p_node = csp.model_share2_root # root node of [M]2 
        while True:
            if p_node.is_leaf_node():
                #print("")
                timestamp_og_2 = (self.timer_og_2.end())
                return (u_node.threshold() + p_node.threshold()) % prime(), timestamp_og_1, timestamp_og_2
            attribute = p_node.attribute()
            #print("attribute, threshold, query value: ", qData.keys()[attribute], ",", (u_node.threshold() + p_node.threshold()) % prime(), ",", 
            #     (csu.qDataShare1[self.attri_list[attribute]] + csp.qDataShare2[self.attri_list[attribute]]) % prime())
            if self._secure_comparison(csu.qDataShare1[self.attri_list[attribute]], ##!! need to check index type
                                        csp.qDataShare2[self.attri_list[attribute]], 
                                        u_node.threshold(), 
                                        p_node.threshold()) == 0:
                p_node = p_node.left_child()
                u_node = self._get_csu_next_child(u_node, True)
            else:
                p_node = p_node.right_child()
                u_node = self._get_csu_next_child(u_node, False)

    #@timed(logger)
    def _secure_comparison(self, x1, x2, y1, y2):# -> int:
        #time.sleep(0.001)

        largest_num = (x1 + x2)%prime() if (x1 + x2)%prime() > (y1 + y2)%prime() else (y1 + y2)%prime()
        upper_bound = int(prime() - largest_num) - 1
        alpha = random.randint(0, upper_bound) #rand_num()
        a1 = rand_num()
        a2 = (alpha - a1) % prime()

        s1 = (x1 + a1) % prime()
        h1 = (y1 + a1) % prime()

        s2 = (x2 + a2) % prime()
        h2 = (y2 + a2) % prime()
        #print("x, y, p | alpha, s1+s2, h1+h2: ", x1 + x2, y1 + y2, prime(), "|", alpha, s1+s2, h1+h2)
        #if (s1+s2) <= (h1+h2):
        if (s1+s2) % prime() <= (h1+h2) % prime():
            return 0
        return 1

    def _get_csu_next_child(self, current, left):# -> Node:
        binary = bin(current.threshold())[2:].zfill(param())
        first = int(binary[:param()//2], 2)
        second = int(binary[param()//2:], 2)

        node = None
        if left:
            node = Node(threshold=pseudo_random_generator(first))
        else:
            node = Node(threshold=pseudo_random_generator(second))
        return node

    ## SecDT plus Ver.1
    def prepare_plusV1(self, mo, csu, csp, root_node, attri_list):
        '''[+Ver.1]'''
        mo.input_model_and_gen_plusV1_shares(root_node, attri_list)
        mo.set_shares_to_two_parties_plusV1(csu, csp)
        return

    def plusV1_classify(self, csu, csp, qData):
        '''[+Ver.1]'''
        self.timer_v1_1.reset("Split query data")
        csu.set_query_data_plusV1(qData)
        csu.send_query_data_to_csp_plusV1(csp)
        
        u_node = csu.root_node
        p_node = csp.model_share2_root # root node of [M]2

        qData_share1_shffled_order, qData_share2_shffled_order = \
            self._matrix_vector_secure_multiplication(len(csu.qDataShare1), csu.qDataShare1, csu.matrix_A_share1, 
                                                    csp.qDataShare2, csp.matrix_A_share2)
        timestamp_v1_1 = (self.timer_v1_1.end("Split query data"))

        self.timer_v1_2.reset("Evaluation once")
        while True:
            if p_node.is_leaf_node():
                timestamp_v1_2 = (self.timer_v1_2.end("Evaluation once"))
                return (u_node.threshold() + p_node.threshold()) % prime(), timestamp_v1_1, timestamp_v1_2
            attribute = p_node.attribute()
            #print("threshold, data value: ", (u_node.threshold() + p_node.threshold()) % prime(), 
            #      (qData_share1_shffled_order[attribute] + qData_share2_shffled_order[attribute]) % prime())
            if self._secure_comparison(qData_share1_shffled_order[attribute], 
                                       qData_share2_shffled_order[attribute], 
                                       u_node.threshold(), 
                                       p_node.threshold()) == 0:
                p_node = p_node.left_child()
                u_node = self._get_csu_next_child(u_node, True)
            else:
                p_node = p_node.right_child()
                u_node = self._get_csu_next_child(u_node, False)

    def _matrix_vector_secure_multiplication(
            self, n, x_share1, m_share1, x_share2, m_share2):# -> list[list, list]:
        ## a = [a]1 + [a]2, b = [b]1 + [b]2, a*b = [a*b]1 + [a*b]2
        ## The initail random value a, b could be generated by Beaver Triple?
        a = 7
        a1 = 3
        a2 = 4
        b = 11
        b1 = 5
        b2 = 6
        ab1 = 33 ## a1 * b1 + a1 * b2
        ab2 = 44 ## a2 * b2 + a2 * b1

        x1 = []
        x2 = []
        for j in range(n):
            temp1 = 0 # CSU
            temp2 = 0 # CSP
            for i in range(n):
                #e1 = (x_share1[i] - a1) % prime()# CSU send e1 to CSP
                #e2 = (x_share2[i] - a2) % prime()# CSP send e2 to CSU
                #e = (e1 + e2) % prime()
                e = (((x_share1[i] + x_share2[i]) % prime()) - a) % prime()
                #p1 = (m_share1[j][i] - b1) % prime() # CSU send p1 to CSP
                #p2 = (m_share2[j][i] - b2) % prime() # CSP send p2 to CSU
                #p = (p1 + p2) % prime()
                p = (((m_share1[j][i] + m_share2[j][i]) % prime()) - b) % prime()
                temp1 = (temp1 + ((ab1 + (((e*b1) % prime()) +  ((p*a1) % prime())    + ((e*p) % prime())      % prime())) % prime())) % prime() # CSU
                temp2 = (temp2 + ((ab2 + (((e*b2) % prime()) + (((p*a2) % prime())     % prime()) % prime())) % prime())) % prime()# CSP
            x1.append(temp1) # CSU
            x2.append(temp2) # CSP
        return np.array(x1), np.array(x2)
    
    ## SecDT plus Ver.2
    def prepare_plusV2(self, mo, csu, csp, root_node, attri_list):
        '''[+Ver.2]'''
        mo.input_model_and_gen_plusV2_shares(root_node, attri_list)
        mo.set_shares_to_two_parties_plusV2(csu, csp)
        return

    def plusV2_classify(self, csu, csp, qData):
        '''[+Ver.2]'''
        #print("plusV2_classify start")
        self.timer_v2_2.reset("Evaluation once")
        u_node = csu.root_node
        p_node = csp.model_share2_root # root node of [M]2

        #print("prime: ", prime())
        timestamp_v2_1 = 0
        while True:
            self.timer_v2_1.reset("Split query data")
            csu.set_query_data_plusV2(qData[:-1])
            csu.send_query_data_to_csp_plusV2(csp)
            timestamp_v2_1 += (self.timer_v2_1.end("Split query data"))
            if p_node.is_leaf_node():
                timestamp_v2_2 = (self.timer_v2_2.end("Evaluation once"))
                return (u_node.threshold() + p_node.threshold()) % prime(), timestamp_v2_1, timestamp_v2_2
            attribute = p_node.attribute()
            #print("csp.qData_A_l: ", csp.qData_A_l)
            #print("attribute, threshold, value: ", attribute, ",", (u_node.threshold() + p_node.threshold()) % prime(), ",", csp.qData_A_l[attribute])
            if self._secure_comparison_V2(csu.l, 
                                       u_node.threshold(), 
                                       csp.qData_A_l[attribute], 
                                       p_node.threshold()) == 0:
                p_node = p_node.left_child()
                u_node = self._get_csu_next_child(u_node, True)
            else:
                p_node = p_node.right_child()
                u_node = self._get_csu_next_child(u_node, False)

    def _secure_comparison_V2(self, l, y1, s, y2):
        '''[+Ver.2]'''
        ##################  'l' maybe out of prime() range  ##################
        h1 = (y1 + l) % prime() # CSU send h1 to CSP
        h = (h1 + y2) % prime()
        if s <= h:
            return 0
        return 1
    
    def prepare_plusVH(self, mo, csu, csp, root_node, attri_list):
        '''[+Ver.H]'''
        mo.input_model_and_gen_plusVH_shares(root_node, attri_list)
        mo.set_shares_to_two_parties_plusVH(csu, csp)
        return
    
    def plusVH_classify(self, csu, csp, qData):
        '''[+Ver.H]'''
        self.timer_vh_2.reset("Evaluation once")
        u_node = csu.root_node
        p_node = csp.model_share2_root # root node of [M]2

        qData_attri_list_hashed = []
        hashed = SHA1.new(b'attri')
        qData_len = len(qData)

        for i in range(qData_len):
            hashed.update(qData.keys()[i].encode())
            qData_attri_list_hashed.append(int.from_bytes(hashed.digest()) % prime())
        self.qData_attri_hash_mapping = list((qData.keys()[i], qData_attri_list_hashed[i]) for i in range(qData_len))
        while True:
            if p_node.is_leaf_node():
                timestamp_vh_2 = (self.timer_vh_2.end("Evaluation once"))
                return (u_node.threshold() + p_node.threshold()) % prime(), timestamp_vh_2

            random.shuffle(self.qData_attri_hash_mapping)
            (qData_attri_list, qData_attri_list_hashed) = zip(*self.qData_attri_hash_mapping)
            qData_suffled = [qData[x] for x in qData_attri_list]
            attriTargeting_v = []
            #print("eta0+eta1: ", (u_node.attribute() + p_node.attribute()) % prime())
            for i in range(qData_len):
                csu.set_query_data_plusVH(qData_attri_list_hashed[i], u_node.attribute())
                csp.set_query_data_plusVH(p_node.attribute())
                attriTargeting_v.append(self._secure_comparison_Vequal(csu.l,
                                        csu.h,
                                        csu.eta0,
                                        csp.eta1))
            #print("attriTargeting_v: ", attriTargeting_v)
            self.attriTargeting_v_share0 = [rand_num() for _ in range(qData_len)]
            self.attriTargeting_v_share1 = [(x - y) % prime() for x, y in zip(attriTargeting_v, self.attriTargeting_v_share0)]
            qDataShare0 = [rand_num() for _ in range(qData_len)]
            qDataShare1 = [(x - y) % prime() for x, y in zip(qData_suffled, qDataShare0)]
            (tq0, tq1) = self.dot_product_triples(qData_len, 
                                                sy.Matrix(self.attriTargeting_v_share0), 
                                                sy.Matrix(self.attriTargeting_v_share1), 
                                                sy.Matrix(qDataShare0), 
                                                sy.Matrix(qDataShare1))
            csu.setTQ(tq0)
            csp.setTQ(tq1)
            if self._secure_comparison(tq0, tq1,
                                       u_node.threshold(), 
                                       p_node.threshold()) == 0:
                #print("left")
                p_node = p_node.left_child()
                u_node = self._get_csu_next_child_with_attri(u_node, True)
            else:
                #print("right")
                p_node = p_node.right_child()
                u_node = self._get_csu_next_child_with_attri(u_node, False)

    def _get_csu_next_child_with_attri(self, current, left):# -> Node:
        binary = bin(current.threshold())[2:].zfill(param())
        first = int(binary[:param()//2], 2)
        second = int(binary[param()//2:], 2)

        binary_attri = bin(current.attribute())[2:].zfill(param())
        first_attri = int(binary_attri[:param()//2], 2)
        second_attri = int(binary_attri[param()//2:], 2)

        node = Node(threshold=pseudo_random_generator(first), 
                    attribute=pseudo_random_generator(first_attri)) if left else Node(threshold=pseudo_random_generator(second), 
                                                                                      attribute=pseudo_random_generator(second_attri))
        return node
    
    def _secure_comparison_Vequal(self, l, h, eta0, eta1):
        '''[+Ver.H]'''
        ##################  'l' maybe out of prime() range  ##################
        h_l = (h + l) % prime() # CSU send h_l to CSP
        eta0_l = (eta0 + l) % prime() # CSU send eta0_l to CSP
        eta_l = (eta0_l + eta1) % prime()
        if eta_l == h_l:
            return 1
        return 0

    def prepare_plusVDP(self, mo, csu, csp, root_node, attri_list):
        '''[+Ver.H]'''
        mo.input_model_and_gen_plusVDP_shares(root_node, attri_list)
        mo.set_shares_to_two_parties_plusVDP(csu, csp)
        return

    def plusVDP_classify(self, csu, csp, qData):
        '''[+Ver.DP]'''
        self.timer_vdp_1.reset("Split query data")
        csu.set_query_data_plusVDP(qData)
        csu.send_query_data_to_csp_plusVDP(csp)
        timestamp_vdp_1 = (self.timer_vdp_1.end("Split query data"))

        self.timer_vdp_2.reset("Evaluation once")
        u_node = csu.root_node
        p_node = csp.model_share2_root # root node of [M]2

        qDataLen = len(qData)
        while True:
            if p_node.is_leaf_node():
                timestamp_vdp_2 = (self.timer_vdp_2.end("Evaluation once"))
                return (u_node.threshold() + p_node.threshold()) % prime(), timestamp_vdp_1, timestamp_vdp_2

            (tq0, tq1) = self.dot_product_triples(qDataLen, 
                                     sy.Matrix(csu.qDataShare0), 
                                     sy.Matrix(csp.qDataShare1), 
                                     sy.Matrix(u_node.attribute()), 
                                     sy.Matrix(p_node.attribute()))
            csu.setTQ(tq0)
            csp.setTQ(tq1)

            if self._secure_comparison(tq0, 
                                       tq1, 
                                       u_node.threshold(), 
                                       p_node.threshold()) == 0:
                #print("left")
                p_node = p_node.left_child()
                u_node = self._get_csu_next_child_with_attri_v(u_node, True)
            else:
                #print("right")
                p_node = p_node.right_child()
                u_node = self._get_csu_next_child_with_attri_v(u_node, False)

    def _get_csu_next_child_with_attri_v(self, current, left):# -> Node:
        binary = bin(current.threshold())[2:].zfill(param())
        first = int(binary[:param()//2], 2)
        second = int(binary[param()//2:], 2)

        attributeShareV1Binary = [bin(x)[2:].zfill(param()) for x in current.attribute()]
        first_attri_v = [int(x[:param()//2], 2) for x in attributeShareV1Binary]
        second_attri_v = [int(x[param()//2:], 2) for x in attributeShareV1Binary]

        node = Node(threshold=pseudo_random_generator(first), 
                    attribute=[pseudo_random_generator(x) for x in first_attri_v]) if left else Node(threshold=pseudo_random_generator(second), 
                                                                                                     attribute=[pseudo_random_generator(x) for x in second_attri_v])
        return node

    def dot_product_triples(self, n, x0, x1, y0, y1):#, Z0=0, Z1=0, X0=[], Y0=[], X1=[], Y1=[]):
        #if n > 0 & len(X0 = 0):
        # X0 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
        # X1 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
        # Y0 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
        # Y1 = sy.Matrix([random.randint(0, 5) for _ in range(n)])
        # T = random.randint(0, 3)
        # Z0 = (X0.dot(Y1) % prime() + T) % prime()
        # Z1 = (X1.dot(Y0) % prime() - T) % prime()

        X0 = sy.Matrix([2 for _ in range(n)])
        X1 = sy.Matrix([2 for _ in range(n)])
        Y0 = sy.Matrix([2 for _ in range(n)])
        Y1 = sy.Matrix([2 for _ in range(n)])
        T = 3
        Z0 = (4*n % prime() + T) % prime()
        Z1 = (4*n % prime() - T) % prime()

        # print("Z", Z0, Z1)
        # print("X: ", X0, X1)
        # print("Y: ", Y0, Y1)
        p0x = (x0 + X0) % prime()
        p0y = (y0 + Y0) % prime()
        p1x = (x1 + X1) % prime()
        p1y = (y1 + Y1) % prime()
        # print("x0 + X0: ", p0x)
        # print("y0 + Y0: ", p0y)
        # print("x1 + X1: ", p1x)
        # print("y1 + Y1: ", p1y)

        z0 = ((x0.dot((y0 + p1y) % prime()) % prime() - Y0.dot(p1x) % prime()) % prime() + Z0) % prime()
        z1 = ((x1.dot((y1 + p0y) % prime()) % prime() - Y1.dot(p0x) % prime()) % prime() + Z1) % prime()
        # print("z0: ", z0)
        # print("z1: ", z1)
        return z0, z1
    
    # def DUORAM_attri_hiding(self, x1, x2,):
    #     att = 1
    #     att0 = 4
    #     att1 = -3
    #     x1 = [0,1,2,3]
    #     x2 = [1,1,1,1]
    #     d0 = x1
    #     d1 = x2
    #     S0 = [0,0,0,0]
    #     S1 = [0,0,0,0]
    #     pi = 3
    #     pi0 = 2
    #     pi1 = 1
    #     epi = [0,0,0,1]
    #     t10 = [0,1,2,3]
    #     t11 = [0,-1,-2,-2]
    #     t20 = [0,0,0,0]
    #     t21 = [0,0,0,1]
    #     t31 = [0,0,0,2]
    #     t30 = [0,0,0,-1]
    #     s_0 = att0 - pi0
    #     s_1 = att1 - pi1
    #     s = s_0 + s_1

    #     T = 7 #random
    #     X0 = np.array([0,1,2,3])
    #     Y0 = np.array([4,3,2,1])
    #     X1 = np.array([4,5,6,7])
    #     Y1 = np.array([3,4,5,6])
    #     Z0 = (X0 @ Y1) + T
    #     Z1 = (X1 @ Y0) - T
        
    #     return