import os

import numpy
from flask import render_template, request, current_app, session

from app.base.constants.BM_CONSTANTS import progress_icon_path, loading_icon_path
from bm.controllers.BaseController import BaseController
from bm.datamanipulation.DataCoderProcessor import DataCoderProcessor
from bm.db_helper.AttributesHelper import get_labels, get_features, get_model_name
from bm.controllers.classification.ClassificationController import ClassificationController
from bm.utiles.Helper import Helper


class ClassificationDirector:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    def predict_values_from_model(self, route_request, ds_goal):
        try:
            features_list = get_features()
            labels_list = get_labels()
            testing_values = []
            opt_param = len(route_request.form)
            all_gategories_values = DataCoderProcessor.get_all_gategories_values()

            if opt_param == 0:
                # response = make_response()
                return render_template('applications/pages/classification/predictevalues.html', features_list=features_list,
                                       labels_list=labels_list,ds_goal = ds_goal,
                                       predicted_value='nothing', testing_values='nothing',
                                       all_gategories_values=all_gategories_values, predicted='Nothing', message='No')

            if opt_param > 0:
                prediction_model = ClassificationController()
                for i in features_list:
                    feature_value = route_request.form.get(i)
                    # final_feature_value = float(feature_value) if feature_value.isnumeric() else feature_value
                    final_feature_value = feature_value
                    testing_values.append(final_feature_value)
                model_name = get_model_name()
                predicted_value = prediction_model.predict_values_from_model(model_name, testing_values)
                # response = make_response()
                return render_template('applications/pages/classification/predictevalues.html',
                                       features_list=features_list,
                                       labels_list=labels_list, ds_goal = ds_goal,
                                       predicted_value=predicted_value, testing_values=testing_values,
                                       all_gategories_values=all_gategories_values, predicted='Yes', message='No')

            return render_template('applications/pages/classification/predictevalues.html',
                                   error=str('Error'), ds_goal = ds_goal,
                                   message="Not able to predict. One or more entered values has not relevant value in your dataset, please enter data from provided dataset",
                                   segment='message')

        except Exception as e:
            return render_template('applications/pages/classification/predictevalues.html',
                                   error=str(e), ds_goal = ds_goal,
                                   message= "Error" + str(e),
                                   segment='message')

    def classify_text_from_model(self, route_request, ds_goal, ds_source):
        try:
            opt_param = len(route_request.form)

            if opt_param == 0:
                # response = make_response()
                return render_template('applications/pages/classification/textpredictevalues.html', ds_goal = ds_goal, ds_source = ds_source,
                                       text_value='', predicted='Nothing', message='No')

            if opt_param > 0:
                input_text = request.form.get('text_value')
                classification_model = ClassificationController()
                text_class = [classification_model.classify_text(input_text)]

                return render_template('applications/pages/classification/textpredictevalues.html',
                                       ds_source= ds_source, ds_goal = ds_goal,
                                       predicted_value=text_class, testing_values=input_text, predicted='Yes', message='No')

            return render_template('applications/pages/classification/textpredictevalues.html',
                                   error=str('Error'), ds_goal = ds_goal, ds_source= ds_source,
                                   message="Not able to predict. One or more entered values has not relevant value in your dataset, please enter data from provided dataset",
                                   segment='message')

        except Exception as e:
            return render_template('applications/pages/classification/predictevalues.html',
                                   error=str(e), ds_goal = ds_goal,
                                   message= "Error" + str(e),
                                   segment='message')

    def prepare_date_files(self):
        # 1. Collect uploaded data files
        # 1. Collect uploaded data files
        return 0

    def create_text_classification_model(self, request):
        try:
            #upload_files = Helper.upload_data_files(folderfiles, mapfile)
            #create_data_bunch = ''
            ds_source = session['ds_source']
            ds_goal = session['ds_goal']
            location_details = {
                'host': request.form.get('location'),
                'username': request.form.get('name'),
                'password': request.form.get('session_token')
            }
            is_local_data = request.form.get('is_local_data') if session['is_local_data'] != 'csv' else session['is_local_data']
            page_url = request.host_url + "predictevalues?t=" + ds_goal + "&s=" + ds_source
            page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"
            classificationcontroller = ClassificationController()
            return_values =  classificationcontroller.run_classification_model(location_details, ds_goal, ds_source, is_local_data)

            return render_template('applications/pages/classification/textmodelstatus.html',
                                   fname=return_values['model_name'],
                                   segment='createmodel',
                                   created_on=return_values['created_on'],
                                   updated_on=return_values['updated_on'],
                                   last_run_time=return_values['last_run_time'],
                                   train_precision=return_values['train_precision'],
                                   train_recall=return_values['train_recall'],
                                   train_f1=return_values['train_f1'],
                                   test_precision=return_values['test_precision'],
                                   test_recall=return_values['test_recall'],
                                   test_f1=return_values['test_f1'],
                                   page_url=page_url, page_embed=page_embed,
                                   most_common=numpy.array(return_values['most_common']),
                                   categories=numpy.array(return_values['categories'])
                                   )

        except Exception as e:
            return render_template('page-501.html', error=e, segment='message')

    def show_model_status(self):
        try:
            model_profile = BaseController.get_model_status()
            page_url = request.host_url + "predictevalues"

            page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"

            return render_template('applications/pages/classification/modelstatus.html',
                                       train_precision=model_profile['train_precision'],
                                       train_recall=model_profile['train_recall'],
                                       train_f1=model_profile['train_f1'],
                                       test_precision=model_profile['test_precision'],
                                       test_recall=model_profile['test_recall'],
                                       test_f1=model_profile['test_f1'],
                                       segment='createmodel', page_url=page_url, page_embed=page_embed,
                                       created_on=model_profile['created_on'],
                                       updated_on=model_profile['updated_on'],
                                       last_run_time=model_profile['last_run_time'],
                                       fname=model_profile['file_name'])

        except Exception as e:
            return render_template('page-501.html', error=e, segment='message')

    def show_text_model_dashboard(self):
        profile = BaseController.get_model_status()
        page_url = request.host_url + "predictevalues?t=10&s=11"
        page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"

        return render_template('applications/pages/classification/textdashboard.html',
                               train_precision=profile['train_precision'],
                               train_recall=profile['train_recall'],
                               train_f1=profile['train_f1'],
                               test_precision=profile['test_precision'],
                               test_recall=profile['test_recall'],
                               test_f1=profile['test_f1'],
                               message='No',
                               fname=profile['model_name'], page_url=page_url, page_embed=page_embed,
                               segment='showdashboard', created_on=profile['created_on'],
                               ds_goal=profile['ds_goal'],
                               updated_on=profile['updated_on'], last_run_time=profile['last_run_time'])