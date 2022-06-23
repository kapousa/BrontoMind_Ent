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
from sklearn.svm import SVR, LinearSVC


class ModelProcessor:
    test_value = ''

    def __init__(self):
        self.test_value = '_'

    def modelselector_backup(self, df: DataFrame, modelfeatures, modellabels):
        number_of_predictions = len(modellabels.axes[1])
        number_of_features = len(modelfeatures.axes[1])
        numberoflabels = modellabels
        numberofrecords = len(df.axes[0])
        numberofrecordsedge = 100000
        if (number_of_predictions == 1):
            labeldatatype = modellabels.dtypes
            if (labeldatatype[0] == np.object):  # Classification
                cls = self.classificationmodelselector(df, modelfeatures, modellabels)
            else:  # Prediction (Regression)
                cls = self.predictionmodelselector(df, modelfeatures, modellabels)
        else: # Clustering
            #cls = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2), n_jobs=-1)
            #cls = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2), n_jobs=-1)
            cls = MultiOutputClassifier(self.classificationmodelselector(df, modelfeatures, modellabels))
        return cls


    def regressionmodelselector(self, df: DataFrame, modellabels):
        number_of_predictions = len(modellabels.axes[1])
        numberofrecords = len(df.axes[0])
        numberofrecordsedge = 100000
        labeldatatype = modellabels.dtypes

        if (numberofrecords <= 50): # no enough data
            return 0;

        if (number_of_predictions == 0):    # Clustering
            cls = KMeans(n_clusters=2, random_state=0)
            return cls

        # if (number_of_predictions == 1 and labeldatatype[0] == np.object):  # Classification
        #     cls = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        #     return cls

        if (number_of_predictions > 0):  # Multi-Output Classification
            cls = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2), n_jobs=-1)
            return cls

        cls = LinearRegression()    # Prediction
        return cls

        # if (number_of_predictions == 1):
        #     if (labeldatatype[0] == np.object):  # Classification
        #         if (numberofrecords < numberofrecordsedge):
        #             try:
        #                 cls = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        #             except:
        #                 cls = SGDClassifier(max_iter=5)
        #         else:
        #             try:
        #                 cls = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
        #             except:
        #                 cls = GaussianNB()
        #     else:  # Prediction (Regression)
        #         if (labeldatatype[0] == np.int64 and numberofrecords < numberofrecordsedge):
        #             cls = SGDRegressor()
        #         elif (labeldatatype[0] == np.int64 and numberofrecords >= numberofrecordsedge):
        #             if (number_of_features < 10):
        #                 cls = Lasso(alpha=1.0)
        #             else:
        #                 try:
        #                     cls = SVR(kernal='linear')
        #                 except:
        #                     cls = SVR(kernal='rbf')
        #         else:
        #             cls = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2), n_jobs=-1)
        # else:
        #     cls = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
        #                                 n_jobs=-1)


    def classificationmodelselector(self, df: DataFrame, modelfeatures, modellabels):
        number_of_predictions = len(modellabels.axes[1])
        number_of_features = len(modelfeatures.axes[1])
        numberoflabels = modellabels
        numberofrecords = len(df.axes[0])
        numberofrecordsedge = 100000
        labeldatatype = modellabels.dtypes
        # Classification
        if (numberofrecords < numberofrecordsedge):
            try:
                cls = SGDClassifier(max_iter=1000, tol=1e-3)
            except:
                cls = SGDClassifier(max_iter=5)
        else:
            try:
                cls = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
            except:
                cls = GaussianNB()
        return cls


    def predictionmodelselector(self, df: DataFrame, modelfeatures, modellabels):
        number_of_predictions = len(modellabels.axes[1])
        number_of_features = len(modelfeatures.axes[1])
        numberoflabels = modellabels
        numberofrecords = len(df.axes[0])
        numberofrecordsedge = 100000
        if (number_of_predictions == 1):
            labeldatatype = modellabels.dtypes
            # Prediction (Regression)
            if ((labeldatatype[0] == np.int64 or labeldatatype[0] == np.float) and numberofrecords < numberofrecordsedge):
                cls = SGDRegressor()
            elif ((labeldatatype[0] == np.int64 or labeldatatype[0] == np.float) and numberofrecords >= numberofrecordsedge):
                if (number_of_features < 10):
                    cls = Lasso(alpha=1.0)
                else:
                    try:
                        cls = SVR(kernal='linear')
                    except:
                        cls = SVR(kernal='rbf')
            else:
                cls = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
                                            n_jobs=-1)
        else:
            cls = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
                                        n_jobs=-1)
        return cls