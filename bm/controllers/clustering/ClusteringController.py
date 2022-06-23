import random
import shutil

from datetime import datetime

import nltk
import numpy
import seaborn as sns
from nltk.corpus import stopwords
from pandas.plotting import scatter_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import bm.core.ModelProcessor
from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix

from app.base.constants.BM_CONSTANTS import plot_zip_locations, pkls_location, output_docs_location, df_location, \
    plot_locations, scalars_location, image_short_path
from bm.controllers.BaseController import BaseController
from bm.core.ClassificationModelProcessor import ClassificationModelProcessor
from bm.core.ModelProcessor import ModelProcessor
from bm.datamanipulation.AdjustDataFrame import encode_prediction_data_frame, \
    decode_predicted_values, deletemodelfiles, encode_one_hot, encode_one_hot_input_features, convert_data_to_sample
from bm.datamanipulation.DataCoderProcessor import DataCoderProcessor
from bm.db_helper.AttributesHelper import get_labels, get_encoded_columns, add_api_details, \
    update_api_details_id
import os
import pickle
from random import randint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import naive_bayes
from app import db
from app.base.db_models.ModelProfile import ModelProfile
from app.base.db_models.ModelForecastingResults import ModelForecastingResults
from app.base.db_models.ModelLabels import ModelLabels
from app.base.db_models.ModelEncodedColumns import ModelEncodedColumns
from app.base.db_models.ModelFeatures import ModelFeatures
from app.base.db_models.ModelAPIDetails import ModelAPIDetails
from bm.datamanipulation.AdjustDataFrame import remove_null_values
from bm.db_helper.AttributesHelper import add_features, add_labels, delete_encoded_columns, get_model_id, \
    encode_testing_features_values, get_features
from bm.utiles.CVSReader import get_only_file_name
from bm.utiles.CVSReader import getcvsheader, get_new_headers_list, reorder_csv_file
import matplotlib.pyplot as plt



class ClusteringController:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']


    def predict_values_from_model(self, model_file_name, testing_values):
        try:
            # ------------------Predict values from the model-------------------------#
            model = pickle.load(open(pkls_location + model_file_name + '_model.pkl', 'rb'))

            # Encode the testing values
            model_id = get_model_id()
            features_list = get_features()
            lables_list = get_labels()
            dcp = DataCoderProcessor()
            testing_values_dic = {}

            testing_values = numpy.array(testing_values)
            # testing_values = testing_values.reshape(1, len(testing_values))
            df_testing_values = pd.DataFrame(testing_values)
            # encode_df_testing_values = df_testing_values if demo == 'DEMO' else dcp.encode_input_values(model_id, testing_values)
            encode_df_testing_values = dcp.encode_input_values(features_list, testing_values)

            # Sclaing testing values
            scalar_file_name = scalars_location + model_file_name + '_scalear.sav'
            s_c = pickle.load(open(scalar_file_name, 'rb'))
            test_x = s_c.transform(encode_df_testing_values)

            predicted_values = model.predict(test_x) #---
            # predicted_values = predicted_values.flatten()
            decoded_predicted_values = dcp.decode_output_values(lables_list, predicted_values)
            print(decoded_predicted_values)
            return decoded_predicted_values

        except  Exception as e:
            print('Ohh -get_model_status...Something went wrong.')
            print(e.with_traceback())
            return [
                'Not able to predict. One or more entered values has not relevant value in your dataset, please enter data from provided dataset']


    def run_classification_model(self, root_path, csv_file_location, predicted_columns, ds_source, ds_goal):
        # ------------------Preparing data frame-------------------------#
        cvs_header = getcvsheader(csv_file_location)
        new_headers_list = get_new_headers_list(cvs_header, predicted_columns)
        reordered_data = reorder_csv_file(csv_file_location, new_headers_list)
        data = reordered_data  # pd.read_csv(csv_file_location)
        model_id = randint(0, 10)

        # Determine features and lables
        features_last_index = len(new_headers_list) - (len(predicted_columns))
        model_features = new_headers_list[0:features_last_index]
        model_labels = predicted_columns

        # 1-Clean the data frame
        data = remove_null_values(data)
        if(len(data) == 0): # No data found after cleaning
            return 0

        dd = data.max(numeric_only=True)
        bb = data.describe()
        print(data.describe())

        # 2- Encode the data frame
        deleteencodedcolumns = delete_encoded_columns()

        data_column_count = len(data.columns)
        testing_values_len = data_column_count - len(predicted_columns)

        # take slice from the dataset, all rows, and cloumns from 0:8
        features_df = data[model_features]
        labels_df = data[model_labels]

        real_x = data.loc[:, model_features]
        real_y = data.loc[:, model_labels]
        dcp = DataCoderProcessor()
        real_x = dcp.encode_features(model_id, real_x)
        real_y = dcp.encode_labels(model_id, real_y)
        encoded_data = pd.concat((real_x, real_y), axis=1, join='inner')
        # real_x = encode_one_hot(model_id, features_df, 'F')  # 2 param (test vales)
        # real_y = encode_one_hot(model_id, labels_df, 'L')  # (predict values)

        training_x, testing_x, training_y, testing_y = train_test_split(real_x, real_y, test_size=0.25, random_state=0)

        # Add standard scalar
        s_c = StandardScaler(with_mean=False)  # test
        training_x = s_c.fit_transform(training_x)
        test_x = s_c.transform(testing_x)
        file_name = get_only_file_name(csv_file_location)
        scalar_file_name = scalars_location + file_name + '_scalear.sav'
        pickle.dump(s_c, open(scalar_file_name, 'wb'))

        # Select proper model
        mp = ClassificationModelProcessor()
        cls = mp.classificationmodelselector(len(encoded_data))
        #cls = # LinearRegression() #MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2), n_jobs=-1)  # KNeighborsRegressor)
        cls.fit(training_x, training_y)

        model_file_name = pkls_location + file_name + '_model.pkl'
        pickle.dump(cls, open(model_file_name, 'wb'))
        y_pred = cls.predict(test_x)

        # Evaluating the Algorithm
        Mean_Absolute_Error = round(
            metrics.mean_absolute_error(numpy.array(testing_y, dtype=object), numpy.array(y_pred, dtype=object)),
            2)  # if not is_classification else 'N/A'
        Mean_Squared_Error = round(metrics.mean_squared_error(testing_y, y_pred), 2)  # if not is_classification else 'N/A'
        Root_Mean_Squared_Error = round(np.sqrt(metrics.mean_squared_error(testing_y, y_pred)),
                                        2)  # if not is_classification else 'N/A'
        c_m = confusion_matrix(testing_y, y_pred)
        c_m = BaseController.get_cm_accurcy(c_m)
        acc = np.array(round(cls.score(training_x, training_y) * 100, 2))

        # Delete old visualization images
        dir = plot_locations
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
        for f in os.listdir(plot_zip_locations):
            os.remove(os.path.join(plot_zip_locations, f))

        # Show prediction
        for i in range(len(model_features)):
            for j in range(len(model_labels)):
                img_prefix = '_' + model_features[i] + '_' +  model_labels[j]
                plot_image_path = os.path.join(plot_locations, get_only_file_name(csv_file_location) + img_prefix +'_plot.png')
                image_path = os.path.join(plot_locations, get_only_file_name(csv_file_location) + img_prefix + '_plot.png')
                if(i ==0 and j ==0):
                    image_db_path = image_short_path + get_only_file_name(csv_file_location) + img_prefix +  '_plot.png'
                sns.pairplot(data, x_vars=model_features[i],
                             y_vars=model_labels[j], size=4, aspect=1, kind='scatter')
                plot_image =  plot_image_path #os.path.join(root_path, 'static/images/plots/', get_only_file_name(csv_file_location) + '_plot.png')
                plt.savefig(plot_image, dpi=300, bbox_inches='tight')
        shutil.make_archive(plot_zip_locations + file_name, 'zip', plot_locations)
        # plt.show()

        # ------------------Predict values from the model-------------------------#
        now = datetime.now()
        all_return_values = {'accuracy': c_m, 'confusion_matrix': c_m, 'plot_image_path': image_db_path, #image_path,
                             'file_name': file_name,
                             'Mean_Absolute_Error': Mean_Absolute_Error,
                             'Mean_Squared_Error': Mean_Squared_Error,
                             'Root_Mean_Squared_Error': Root_Mean_Squared_Error,
                             'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                             'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                             'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S")}

        # Add model profile to the database
        modelmodel = {'model_id': model_id,
                      'model_name': file_name,
                      'user_id': 1,
                      'model_headers': str(cvs_header)[1:-1],
                      'prediction_results_accuracy': str(c_m),
                      'mean_absolute_error': str(Mean_Absolute_Error),
                      'mean_squared_error': str(Mean_Squared_Error),
                      'root_mean_squared_error': str(Root_Mean_Squared_Error),
                      'plot_image_path': image_db_path,
                      'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                      'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                      'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S"),
                      'ds_source': ds_source,
                      'ds_goal': ds_goal}
        model_model = ModelProfile(**modelmodel)
        # Delete current profile
        model_model.query.filter().delete()
        db.session.commit()
        # Add new profile
        db.session.add(model_model)
        db.session.commit()

        # Add features, labels, and APIs details
        add_features_list = add_features(model_id, model_features)
        add_labels_list = add_labels(model_id, model_labels)
        api_details_id = random.randint(0, 22)
        api_details_list = add_api_details(model_id, api_details_id, 'v1')
        api_details_list = update_api_details_id(api_details_id)
        # db.session.commit()
        # db.session.expunge_all()
        # db.close_all_sessions

        # APIs details and create APIs document

        convert_data_to_sample(csv_file_location, 5)
        return all_return_values


# b = run_demo_model1(root_path, 'diabetes.csv', ['Age'], '1', '2')