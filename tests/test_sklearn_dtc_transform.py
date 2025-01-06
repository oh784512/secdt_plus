import unittest
from secdtplus import sklearn_DTC_transform
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.model_selection import train_test_split
import pandas as pd
from secdtplus import preprocessing

class TestTransform(unittest.TestCase):
    def test_init(self):
        names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label']
        data_orig = pd.read_csv("data/heart-disease/processed.cleveland.data", header=None, names=names, index_col=False)
        data = preprocessing.transform3(data_orig)

        margin = int(len(data)*0.8)
        trainingData = data[:margin].reset_index(drop=True)
        testingData = data[margin:].reset_index(drop=True)
        
        x = data.iloc[:,:-1]
        y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(x, y)

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        
        modelTreeRootNode = sklearn_DTC_transform.transform(model.tree_.feature, model.tree_.threshold, model.tree_.value, model.classes_)
        self.dfs(modelTreeRootNode, model.tree_)

    
    def dfs(self, transformed_modelTreeRootNode, sk_modeltree):
        
        self.assertEqual()