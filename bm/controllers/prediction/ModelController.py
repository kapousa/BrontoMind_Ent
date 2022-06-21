import random

from datetime import datetime

import nltk
import numpy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score

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
from matplotlib import pyplot as plt
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



class ModelController:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']


def saveDSFile(self):
    return 'file uploaded successfully'


pkls_location = 'pkls/'
scalars_location = 'scalars/'
plot_locations = 'app/static/images/plots/'
image_short_path = 'static/images/plots/'
df_location = 'app/data/'
image_location = 'app/'
root_path = '../app/'
output_docs_location = 'app/base/output_docs/'
docs_templates_folder = 'docs_templates/'
output_docs = 'app/base/output_docs/'


def run_prediction_model(root_path, csv_file_location, predicted_columns, ds_source, ds_goal, demo):
    if demo == 'DEMO':
        return run_demo_model(root_path, csv_file_location, predicted_columns, ds_source, ds_goal)
    else:
        return run_prod_model(root_path, csv_file_location, predicted_columns, ds_source, ds_goal)


def run_prod_model(root_path, csv_file_location, predicted_columns, ds_source, ds_goal):
    # ------------------Preparing data frame-------------------------#
    cvs_header = getcvsheader(csv_file_location)
    new_headers_list = get_new_headers_list(cvs_header, predicted_columns)
    reordered_data = reorder_csv_file(csv_file_location, new_headers_list)
    data = pd.read_csv(csv_file_location)
    nltk.download("stopwords")
    stopset = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
    model_id = randint(0, 10)

    # Determine features and lables
    features_last_index = len(new_headers_list) - (len(predicted_columns))
    model_features = new_headers_list[0:features_last_index]
    model_labels = predicted_columns

    # 1-Clean the data frame
    data = remove_null_values(data)

    # take slice from the dataset, all rows, and cloumns from 0:8
    data_column_count = len(data.columns)
    testing_values_len = data_column_count - len(predicted_columns)
    features_df = data[model_features]
    labels_df = data[model_labels]

    # Check if the dataset is only numric
    features_dt_array = np.array(np.where(features_df.dtypes == np.object))
    features_dt_array = features_dt_array.flatten()
    lables_dt_array = np.array(np.where(labels_df.dtypes == np.object))
    lables_dt_array = lables_dt_array.flatten()
    is_classification = True if ((len(features_dt_array) != 0) and (len(lables_dt_array) != 0)) else False  # Prediction

    real_x = data.loc[:, model_features]
    real_y = data.loc[:, model_labels]
    # real_x_m = vectorizer.fit_transform(data)
    training_x, testing_x, training_y, testing_y = train_test_split(real_x, real_y, test_size=0.20, random_state=42)

    clf = naive_bayes.MultinomialNB()
    clf.fit(training_x, training_y)
    accuracy_score(testing_y, clf.predict(testing_x)) * 100
    clf = naive_bayes.MultinomialNB()
    clf.fit(real_x, real_y)

    accuracy_score(testing_y, clf.predict(testing_x)) * 100
    file_name = get_only_file_name(csv_file_location)
    model_file_name = pkls_location + file_name + 'nlp_model.pkl'
    pickle.dump(clf, open(model_file_name, 'wb'))

    # ------------------Predict values from the model-------------------------#
    now = datetime.now()
    all_return_values = {'accuracy': 'acc', 'confusion_matrix': 'c_m', 'plot_image_path': 'plot_image_path',
                         'file_name': file_name,
                         'Mean_Absolute_Error': 'Mean_Absolute_Error',
                         'Mean_Squared_Error': 'Mean_Squared_Error',
                         'Root_Mean_Squared_Error': 'Root_Mean_Squared_Error',
                         'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                         'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                         'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S")}

    # Add model profile to the database
    modelmodel = {'model_id': model_id,
                  'model_name': file_name,
                  'user_id': 1,
                  'model_headers': str(cvs_header)[1:-1],
                  'prediction_results_accuracy': str('acc'),
                  'mean_absolute_error': str('Mean_Absolute_Error'),
                  'mean_squared_error': str('Mean_Squared_Error'),
                  'root_mean_squared_error': str('Root_Mean_Squared_Error'),
                  'plot_image_path': 'plot_image_path',
                  'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                  'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                  'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S"),
                  'ds_source': ds_source}
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
    db.session.commit()

    return all_return_values


def predict_values_from_model(model_file_name, testing_values):
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

        predicted_values = model.predict(test_x)
        # predicted_values = predicted_values.flatten()
        decoded_predicted_values = dcp.decode_output_values(lables_list, predicted_values)
        print(decoded_predicted_values)
        return decoded_predicted_values

    except  Exception as e:
        print('Ohh -get_model_status...Something went wrong.')
        print(e.with_traceback())
        return [
            'Not able to predict. One or more entered values has not relevant value in your dataset, please enter data from provided dataset']


def predict_values_from_model_with_encode(model_file_name, testing_values): # Not used
    # ------------------Predict values from the model-------------------------#
    model = pickle.load(open(pkls_location + model_file_name + '_model.pkl', 'rb'))

    # Encode the testing values
    model_id = get_model_id()
    features_list = get_features()
    labels_list = get_labels()
    encoded_labels_list = get_encoded_columns('L')

    testing_values_dic = {}
    for i in range(len(features_list)):
        testing_values_dic[features_list[i]] = testing_values[i]
    reshaped_testing_values = np.reshape(testing_values, (1, len(testing_values)))
    reshaped_testing_values = reshaped_testing_values.flatten()
    encoded_testing_values = [encode_prediction_data_frame(reshaped_testing_values, 'F')]
    df_testing_values = pd.DataFrame(encoded_testing_values)
    predicted_values = model.predict(df_testing_values)
    predicted_values = predicted_values.flatten()
    print(predicted_values)
    decoded_predicted_values = decode_predicted_values(model_id, predicted_values, labels_list, encoded_labels_list)

    return decoded_predicted_values


def run_demo_model(root_path, csv_file_location, predicted_columns, ds_source, ds_goal):
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
    cls = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
                                n_jobs=-1)  # KNeighborsRegressor)
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
    c_m = ''
    acc = np.array(round(cls.score(training_x, training_y) * 100, 2))

    # Delete old visualization images
    dir = plot_locations
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    # Show prediction
    plot_image_path = os.path.join(plot_locations, get_only_file_name(csv_file_location) + '_plot.png')
    image_path = os.path.join(plot_locations, get_only_file_name(csv_file_location) + '_plot.png')
    image_db_path = image_short_path + get_only_file_name(csv_file_location) + '_plot.png'
    # if not is_classification:
    plt.scatter(testing_y, y_pred, color="red")
    x = [np.min(testing_y), np.max(testing_y)]
    y = [np.min(y_pred), np.max(y_pred)]
    plt.plot(x, y, color="#52b920", label='Regression Line')
    plt.title("Testing model vs Prediction values")
    plt.xlabel("Tested values")
    plt.ylabel("Predicted values")
    plot_image = plot_image_path  # os.path.join(root_path, 'static/images/plots/', get_only_file_name(csv_file_location) + '_plot.png')
    plt.savefig(plot_image, dpi=300, bbox_inches='tight')
    # plt.show()

    # ------------------Predict values from the model-------------------------#
    now = datetime.now()
    all_return_values = {'accuracy': acc, 'confusion_matrix': c_m, 'plot_image_path': image_path,
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
                  'prediction_results_accuracy': str(acc),
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


def get_model_status():
    try:
        model_profile_row = ModelProfile.query.all()
        model_profile = {}

        for profile in model_profile_row:
            model_profile = {'model_id': profile.model_id,
                             'model_name': profile.model_name,
                             'prediction_results_accuracy': str(profile.prediction_results_accuracy),
                             'mean_absolute_error': str(profile.mean_absolute_error),
                             'mean_squared_error': str(profile.mean_squared_error),
                             'root_mean_squared_error': str(profile.root_mean_squared_error),
                             'plot_image_path': profile.plot_image_path,
                             'created_on': profile.created_on,
                             'updated_on': profile.updated_on,
                             'last_run_time': profile.last_run_time,
                             'ds_sorce': profile.ds_source,
                             'ds_goal': profile.ds_goal,
                             'mean_percentage_error': profile.mean_percentage_error,
                             'mean_absolute_percentage_error': profile.mean_absolute_percentage_error,
                             'depended_factor': profile.depended_factor,
                             'forecasting_category': profile.forecasting_category}
            print(model_profile)
        return model_profile
    except  Exception as e:
        print('Ohh -get_model_status...Something went wrong.')
        print(e)
        return 0


def delet_model():
    try:
        ModelEncodedColumns.query.filter().delete()
        ModelFeatures.query.filter().delete()
        ModelLabels.query.filter().delete()
        ModelAPIDetails.query.filter().delete()
        ModelProfile.query.filter().delete()
        ModelForecastingResults.query.filter().delete()
        db.session.commit()

        # Delete old model files
        delete_model_files = deletemodelfiles(scalars_location, pkls_location, output_docs_location, df_location)

        return 1
    except Exception as e:
        print('Ohh -delet_models...Something went wrong.')
        print(e)
        return 0
