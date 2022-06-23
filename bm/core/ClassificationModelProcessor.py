#  Copyright (c) 2021. Slonos Labs. All rights Reserved.
import numpy as np
from pandas import DataFrame
from sklearn.cluster import MeanShift, KMeans
from sklearn.linear_model import SGDRegressor, SGDClassifier, LinearRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVC, SVC


class ClassificationModelProcessor:
    test_value = ''

    def __init__(self):
        self.test_value = '_'

    def classificationmodelselector(self, numberofrecords):
        numberofrecordsedge = 100000

        # Classification
        if (numberofrecords < numberofrecordsedge):
            try:
                cls = LinearSVC(verbose=0)
                LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                          verbose=0)
                return cls
            except:
                try:
                    cls = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
                    return cls
                except:
                    try:
                        cls = SVC(probability=True, decision_function_shape='ovo', kernel='rbf', gamma=0.0078125, C=8)
                        return cls
                    except Exception as e:
                        print(e)
                        return 0

        if (numberofrecords >= numberofrecordsedge):
            try:
                cls = SGDClassifier(max_iter=1000, tol=0.01)
                return cls
            except Exception as e:
                print(e)
                return 0
        return 0
