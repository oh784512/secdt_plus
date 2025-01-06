from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sympy as sy
import argparse
import logging

from preprocessing import transform1, transform2, transform3, transform4, transform5, transform6 
#1:nursery.data #2:weatherAUS.csv #3:heart-disease/processed.cleveland.data #4:bank.csv #5:malware.csv #6:學期成績.csv
from sklearn_DTC_transform import transform
from entity2_secdt import ModelOwner, CloudServiceProvider, CloudServiceUser
from entity2_protocol import Protocol
from entity2_secdtplus_ver1 import ModelOwnerVer1, CloudServiceProviderVer1, CloudServiceUserVer1
from entity2_secdtplus_ver2 import ModelOwnerVer2, CloudServiceProviderVer2, CloudServiceUserVer2
from entity2_secdtplus_verh import ModelOwnerVerH, CloudServiceProviderVerH, CloudServiceUserVerH
from entity2_secdtplus_verdp import ModelOwnerVerDP, CloudServiceProviderVerDP, CloudServiceUserVerDP
from structure2 import Timer, Node
from utility2 import size, tree_info
from secure import prime

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import _tree
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz 

def setpriority(pid=None,priority=1):
    """ Set The Priority of a Windows Process.  Priority is a value between 0-5 where
        2 is normal priority.  Default sets the priority of the current
        python process but can take any valid process ID. """
        
    import win32api,win32process,win32con
    
    priorityclasses = [win32process.IDLE_PRIORITY_CLASS,
                       win32process.BELOW_NORMAL_PRIORITY_CLASS,
                       win32process.NORMAL_PRIORITY_CLASS,
                       win32process.ABOVE_NORMAL_PRIORITY_CLASS,
                       win32process.HIGH_PRIORITY_CLASS,
                       win32process.REALTIME_PRIORITY_CLASS]
    if pid == None:
        pid = win32api.GetCurrentProcessId()
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
    win32process.SetPriorityClass(handle, priorityclasses[priority])

## traversed to print tree
# def childNodes(i):
#      return (2*i)+1, (2*i)+2

# def traversed(a, i=0, d = 0):
#     if i >= len(a):
#         return 
#     l, r =  childNodes(i)
#     traversed(a, r, d = d+1)
#     if a[i] != -2:
#         print("   "*d + str(a[i]))
#     else:
#         print("   "*d + "'")
#     traversed(a, l, d = d+1)

## 
# def tree_to_code(tree, feature_names):
#     tree_ = tree.tree_
#     feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
#     print("def tree({}):".format(", ".join(feature_names)))

#     def recurse(node, depth):
#         indent = "  " * depth
#         if tree_.feature[node] != _tree.TREE_UNDEFINED:
#             name = feature_name[node]
#             threshold = tree_.threshold[node]
#             print("{}if {} <= {}:".format(indent, name, threshold))
#             recurse(tree_.children_left[node], depth + 1)
#             print("{}else:  # if {} > {}".format(indent, name, threshold))
#             recurse(tree_.children_right[node], depth + 1)
#         else:
#             print("{}return {}".format(indent, tree_.value[node]))
#     recurse(0, 1)

parser = argparse.ArgumentParser(description="SecDT+ demo")
parser.add_argument('-s', '--data', type=int, choices=range(1, 7), default=3, help="Choose data set to test classifier.")
#parser.add_argument('-v', '--version', type=int, choices=range(1, 2), help="Choose SecDT+ version")
args = parser.parse_args()


def get_data():
    data = 0
    if args.data== 1:
        names = ['parents', 'has_nurs', 'form', 'children',
                'housing', 'finance', 'social', 'health', 'label']
        data_orig = pd.read_csv('data/nursery/nursery.data', header=None, names=names, index_col=False)
        data = transform1(data_orig)
    if args.data== 2:
        data_orig = pd.read_csv('data/weather/weatherAUS.csv', index_col=False)
        data = transform2(data_orig)
    if args.data== 3:
        names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label']
        data_orig = pd.read_csv("data/heart-disease/processed.cleveland.data", header=None, names=names, index_col=False)
        data = transform3(data_orig)
    if args.data== 4:
        data_orig = pd.read_csv('data/bank/bank.csv', index_col=False)
        data = transform4(data_orig)
    if args.data== 5:
        data_orig = pd.read_csv('data/malware/malware.csv', index_col=False)
        data = transform5(data_orig)
    if args.data== 6:
        data_orig = pd.read_csv('data/學期成績/學期成績.csv', index_col=False)
        data = transform6(data_orig)
        # case 7:
        # case 8:
    return data

def copyTree(node):
    if node.is_leaf_node():
        return Node(attribute=node.attribute(), threshold=node.threshold(), is_leaf_node=True)
    leftNode = copyTree(node.left_child())
    rightNode = copyTree(node.right_child())
    return Node(attribute=node.attribute(), threshold=node.threshold(), left_child=leftNode, right_child=rightNode)

if __name__ == "__main__":
    #FORMAT = '%(filename)s[line:%(lineno)d] %(levelname)s => %(message)s'
    #logging.basicConfig(level=logging.DEBUG, filemode='secdtplus.log', format=FORMAT)

    timestamp1 = []
    timestamp2 = []
    csp_size_alloc=[]
    csu_size_alloc=[]
    split_time_alloc=[]
    eval_time_alloc=[]
    communication_size_alloc=[]

    model = DecisionTreeClassifier()
    #data = pd.read_csv('data/學期成績/學期成績.csv')
    
    #data_orig = pd.read_csv('data/bank/bank.csv')
    #data = transform4(data_orig)
    
    #data = pd.read_csv('data/malware/malware.csv')
    
    #data = pd.read_csv('data/weather/weatherAUS.csv')

    # names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    #          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label']
    # data_orig = pd.read_csv("data/heart-disease/processed.cleveland.data", header=None, names=names, index_col=False)
    # data = transform3(data_orig)
    data = get_data()

    margin = int(len(data)*0.8)
    trainingData = data[:margin].reset_index(drop=True)
    testingData = data[margin:].reset_index(drop=True)
    
    x = data.iloc[:,:-1]
    y = data.iloc[:, -1]
    print("x:\n", x)
    print("y:\n", y)
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    #print("1", classification_report(y_test, pred))
    #print("2", confusion_matrix(y_test, pred))
    plot_tree(model)
    
    #print(type(model))
    #print("model.__dict__: ", model.__dict__)
    
    print("model.tree_: ", model.tree_)
    #print("model.tree_.value: \n", model.tree_.value)

## show all threshold and feature of internal nodes
    # print("threshold: ")
    # print("threshold len: ")
    # print("feature: ")
    # print("feature len: ")
    
## test
    dot_data = tree.export_graphviz(model, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("iris")

    modelTreeRootNode = transform(model.tree_.feature, model.tree_.threshold, model.tree_.value, model.classes_)
    modelTreeRootNode_v1 = copyTree(modelTreeRootNode)
    modelTreeRootNode_v2 = copyTree(modelTreeRootNode)
    modelTreeRootNode_h = copyTree(modelTreeRootNode)
    modelTreeRootNode_dp = copyTree(modelTreeRootNode)
    
    #print(modelTreeRootNode, modelTreeRootNode.__dict__)

    p = Protocol()

    # #tesssssssssssssssssst
    # #a=[6,8,10] -> [10,6,8]
    # #m = [[0, 0, 1],
    # #     [1, 0, 0],
    # #     [0, 1, 0]]
    # a1 = [1, 2, 3]
    # a2 = [5, 6, 7]
    # m1 = [[-1, 0, -3],
    #       [0, 0, 1],
    #       [-2, 1, -1]]
    # m2 = [[1, 0, 4],
    #       [1, 0, -1],
    #       [2, 0, 1]]

    # am1, am2 = p._matrix_vector_secure_multiplication(3, a1, m1, a2, m2)
    # print("TEST: ", am1, am2)

    mo = ModelOwner()
    csp = CloudServiceProvider()
    csu = CloudServiceUser()

    moV1 = ModelOwnerVer1()
    cspV1 = CloudServiceProviderVer1()
    csuV1 = CloudServiceUserVer1()

    moV2 = ModelOwnerVer2()
    cspV2 = CloudServiceProviderVer2()
    csuV2 = CloudServiceUserVer2()

    moVH = ModelOwnerVerH()
    cspVH = CloudServiceProviderVerH()
    csuVH = CloudServiceUserVerH()

    moVDP = ModelOwnerVerDP()
    cspVDP = CloudServiceProviderVerDP()
    csuVDP = CloudServiceUserVerDP()

    p.prepare(mo, csu, csp, modelTreeRootNode)
    p.set_attri_list(model.feature_names_in_)

    original_classication_result_class_list = []
    plusV1_classication_result_class_list = []
    plusV2_classication_result_class_list = []
    plusVH_classication_result_class_list = []
    plusVDP_classication_result_class_list = []
    
    attri_num = len(trainingData.loc[0][:-1])
    classification_times = 150
    print("==========Start original classification ", classification_times, " times...")
    for i in range(classification_times):
        #timer1.reset("Split query data")
        # csu.set_query_data(trainingData.loc[i])
        # csu.send_query_data_to_csp(csp)
        #timestamp1.append(timer1.end())
        original_classication_result_class, tstmp1, tstmp2 = p.classify(csu, csp, trainingData.loc[i])
        original_classication_result_class_list.append(original_classication_result_class)
        timestamp1.append(tstmp1)
        timestamp2.append(tstmp2)
        #timer2.reset("Evaluation once")
        #protocol(csu, csp, csu.node, csp.node)
        #p.start(csu, csp, csu.root_node, csp.model_share2_root)
        #timestamp2.append(timer2.end())
    split_time_alloc.append(round(np.mean(timestamp1),2))
    eval_time_alloc.append(round(np.mean(timestamp2),2))
    nodes, depth = tree_info(mo.get_model())
    csu_size=size(csu.root_node) * depth + size(csu.seed) + size(csu.qData) + size(csu.qDataShare1)+size(csu.qDataShare2)
    csu_size_alloc.append(round(csu_size/1024,2))
    csp_size=size(csp.model_share2_root) + size(csp.qDataShare2)
    csp_size_alloc.append(round(csp_size/1024,2)) 
    result_size = np.average([size(x) for x in original_classication_result_class_list])
    communication_size = size(csu.qDataShare2)+size(csp.model_share2_root.attribute())
    communication_size_alloc.append(round(communication_size/1024,2)) 
    #print(split_time_alloc,eval_time_alloc)
    #print(csu_size_alloc,csp_size_alloc)
    print("tree node count, depth:", nodes, depth)
    print("==========End original classification.")

    timestamp_v1_1 = []
    timestamp_v1_2 = []
    # csp_size_alloc=[]
    # csu_size_alloc=[]
    # split_time_alloc=[]
    # eval_time_alloc=[]
    # communication_size_alloc=[]

    ###### Ver.1
    p.prepare_plusV1(moV1, csuV1, cspV1, modelTreeRootNode_v1, model.feature_names_in_)
    print("==========Start +v1 classification ", classification_times, " times...")
    for i in range(classification_times):
        plusV1_classication_result_class, tstmp1, tstmp2 = p.plusV1_classify(csuV1, cspV1, trainingData.loc[i])
        plusV1_classication_result_class_list.append(plusV1_classication_result_class)
        timestamp_v1_1.append(tstmp1)
        timestamp_v1_2.append(tstmp2)

    # for i in range(classification_times):
    #     mark = '' if original_classication_result_class_list[i] == plusV1_classication_result_class_list[i] else '*'
    #     print("original_classication_result_class, plusV1_classication_result_class: ", 
    #           original_classication_result_class_list[i], plusV1_classication_result_class_list[i], mark)
    split_time_alloc.append(round(np.mean(timestamp_v1_1),2))
    eval_time_alloc.append(round(np.mean(timestamp_v1_2),2))
    nodes, depth = tree_info(moV1.get_model())
    csu_size=size(csuV1.root_node) * depth + size(csuV1.seed) + size(csuV1.qData) + size(csuV1.qDataShare1)+size(csuV1.qDataShare2)
    csu_size_alloc.append(round(csu_size/1024,2))
    csp_size=size(cspV1.model_share2_root) + size(cspV1.qDataShare2)
    csp_size_alloc.append(round(csp_size/1024,2)) 
    result_size = np.average([size(x) for x in plusV1_classication_result_class_list])
    communication_size = size(csu.qDataShare2)+size(prime()/2)*4*attri_num*attri_num+(size(csp.model_share2_root.attribute())+(size(prime()/2))*3)*depth + result_size
    communication_size_alloc.append(round(communication_size/1024,2)) 
    #print(split_time_alloc, eval_time_alloc)
    #print(csu_size_alloc,csp_size_alloc)
    print("tree node count, depth:", nodes, depth)
    print("==========End +v1 classification.")

    timestamp_v2_1 = []
    timestamp_v2_2 = []
    # csp_size_alloc=[]
    # csu_size_alloc=[]
    # split_time_alloc=[]
    # eval_time_alloc=[]
    # communication_size_alloc=[]

    ###### Ver.2
    p.prepare_plusV2(moV2, csuV2, cspV2, modelTreeRootNode_v2, model.feature_names_in_)
    print("==========Start +v2 classification ", classification_times, " times...")
    for i in range(classification_times):
        plusV2_classication_result_class, tstmp1, tstmp2 = p.plusV2_classify(csuV2, cspV2, trainingData.loc[i])
        plusV2_classication_result_class_list.append(plusV2_classication_result_class)
        timestamp_v2_1.append(tstmp1)
        timestamp_v2_2.append(tstmp2 - tstmp1)

    # for i in range(classification_times):
    #     mark = '' if original_classication_result_class_list[i] == plusV2_classication_result_class_list[i] else '*'
    #     print("original_classication_result_class, plusV2_classication_result_class: ", 
    #           original_classication_result_class_list[i], plusV2_classication_result_class_list[i], mark)
    split_time_alloc.append(round(np.mean(timestamp_v2_1),2))
    eval_time_alloc.append(round(np.mean(timestamp_v2_2),2))
    nodes, depth = tree_info(moV2.get_model())
    csu_size=size(csuV2.root_node) * depth + size(csuV2.seed)+size(csuV2.qData)+size(csuV2.qData_l_S)+size(csuV2.matrix_S)
    csu_size_alloc.append(round(csu_size/1024,2))
    csp_size=size(cspV2.model_share2_root) + size(cspV2.qData_A_l)+size(cspV2.matrix_K)
    csp_size_alloc.append(round(csp_size/1024,2))
    result_size = np.average([size(x) for x in plusV2_classication_result_class_list])
    communication_size = (size(csuV2.qData_l_S)+(size(prime()/2))*3)*depth + result_size
    communication_size_alloc.append(round(communication_size/1024,2))
    #print(split_time_alloc, eval_time_alloc)
    #print(csu_size_alloc,csp_size_alloc)
    print("tree node count, depth:", nodes, depth)
    print("==========End +v2 classification.")


    timestamp5 = []
    # csp_size_alloc=[]
    # csu_size_alloc=[]
    # split_time_alloc=[]
    # eval_time_alloc=[]
    # communication_size_alloc=[]

    ###### Ver.Hash-based
    p.prepare_plusVH(moVH, csuVH, cspVH, modelTreeRootNode_h, model.feature_names_in_)
    print("==========Start +vH classification ", classification_times, " times...")
    for i in range(classification_times):
        plusVH_classication_result_class, tstmp5 = p.plusVH_classify(csuVH, cspVH, trainingData.loc[i][:-1])
        plusVH_classication_result_class_list.append(plusVH_classication_result_class)
        timestamp5.append(tstmp5)

    # for i in range(classification_times):
    #     mark = '' if original_classication_result_class_list[i] == plusVH_classication_result_class_list[i] else '*'
    #     print("original_classication_result_class, plusVH_classication_result_class: ", 
    #           original_classication_result_class_list[i], plusVH_classication_result_class_list[i], mark)
    split_time_alloc.append(-1)
    eval_time_alloc.append(round(np.mean(timestamp5),2))
    nodes, depth = tree_info(moVH.get_model())
    csu_size=size(csuVH.root_node) * depth + size(csuVH.seed) + size(csuVH.l) + size(csuVH.h) + size(csuVH.eta0)+size(csuVH.tq0)
    csu_size_alloc.append(round(csu_size/1024,2))
    csp_size=size(cspVH.model_share2_root) + size(cspVH.eta1)+size(cspVH.tq1)
    csp_size_alloc.append(round(csp_size/1024,2))
    result_size = np.average([size(x) for x in plusVH_classication_result_class_list])
    communication_size = (size(csuVH.h)+size(csuVH.eta0)+size(p.attriTargeting_v_share0)+(size(prime()/2))*3)*depth + result_size
    communication_size_alloc.append(round(communication_size/1024,2))
    #print(split_time_alloc, eval_time_alloc)
    #print(csu_size_alloc,csp_size_alloc)
    print("tree node count, depth:", nodes, depth)
    print("==========End +vH classification.")


    timestamp_vdp_1 = []
    timestamp_vdp_2 = []
    # csp_size_alloc=[]
    # csu_size_alloc=[]
    # split_time_alloc=[]
    # eval_time_alloc=[]
    # communication_size_alloc=[]

    ###### Ver.Dot-Product
    p.prepare_plusVDP(moVDP, csuVDP, cspVDP, modelTreeRootNode_dp, model.feature_names_in_)
    print("==========Start +vDP classification ", classification_times, " times...")
    for i in range(classification_times):
        plusVDP_classication_result_class, tstmp1, tstmp2 = p.plusVDP_classify(csuVDP, cspVDP, trainingData.loc[i][:-1])
        plusVDP_classication_result_class_list.append(plusVDP_classication_result_class)
        timestamp_vdp_1.append(tstmp1)
        timestamp_vdp_2.append(tstmp2)

    # for i in range(classification_times):
    #     mark = '' if original_classication_result_class_list[i] == plusVDP_classication_result_class_list[i] else '*'
    #     print("original_classication_result_class, plusVDP_classication_result_class: ", 
    #           original_classication_result_class_list[i], plusVDP_classication_result_class_list[i], mark)
    split_time_alloc.append(round(np.mean(timestamp_vdp_1),2))
    eval_time_alloc.append(round(np.mean(timestamp_vdp_2),2))
    nodes, depth = tree_info(moVDP.get_model())
    csu_size=size(csuVDP.root_node) * depth +size(csuVDP.seed)+size(csuVDP.seed_attri_v)+size(csuVDP.qData)+size(csuVDP.qDataShare0)+size(csuVDP.qDataShare1)+size(csuVDP.tq0)
    csu_size_alloc.append(round(csu_size/1024,2))
    csp_size=size(cspVDP.model_share2_root) + size(cspVDP.qDataShare1)+size(cspVDP.tq1)
    csp_size_alloc.append(round(csp_size/1024,2))
    result_size = np.average([size(x) for x in plusVDP_classication_result_class_list])
    communication_size=size(csuVDP.qDataShare0)*depth+(size(
        sy.Matrix(csuVDP.qDataShare0))+size(
            sy.Matrix(cspVDP.qDataShare1))+size(
                sy.Matrix(csuVDP.root_node.attribute()))+size(
                    sy.Matrix(cspVDP.model_share2_root.attribute())))*depth + (size(prime()/2))*3*depth + result_size
    communication_size_alloc.append(round(communication_size/1024,2))
    #print(split_time_alloc, eval_time_alloc,"=>split_time_alloc, eval_time_alloc (unit: MB)")
    #print(csu_size_alloc,csp_size_alloc,"=>csu_size_alloc, csp_size_alloc (unit: MB)")
    #print(communication_size_alloc, "=>communication_size_alloc (unit: MB)")

    print("tree node count, depth:", nodes, depth)
    print("==========End +vDP classification.")

    print("attribute numbers: ", attri_num)

    schemeNames = ["org","v1","v2","vh","vdp"]
    print('{:4s}|{:10s}|{:9s}|{:12s}|{:12s}|{:13s}'.format("", "split_time", "eval_time", "csu_capacity", "csp_capacity", "communication"))
    for schemeName, split_time, eval_time, csu_capacity, csp_capacity, communication in zip(schemeNames, split_time_alloc,eval_time_alloc,csu_size_alloc,csp_size_alloc,communication_size_alloc):
        print('{:4s}|{:10g}|{:9g}|{:12g}|{:12g}|{:13g}'.format(schemeName, split_time, eval_time, csu_capacity, csp_capacity, communication))

    #print("First ending.")