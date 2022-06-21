# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import os
import io
import shutil
import sys
import traceback

import pandas as pd
import numpy
from flask import Markup
import json

from app.base.db_models.ModelForecastingResults import ModelForecastingResults
from app.base.db_models.ModelLabels import ModelLabels
from app.base.db_models.ModelProfile import ModelProfile
from app.base.db_models.ModelAPIDetails import ModelAPIDetails
from flask import redirect, send_file, url_for, Response, Flask, \
    current_app
from matplotlib.backends.backend_template import FigureCanvas
from werkzeug.utils import secure_filename

from app import login_manager

from bm.apis.v1.APIHelper import APIHelper
from bm.controllers.mlforecasting.MLForecastingController import MLForecastingController
from bm.core.DocumentProcessor import DocumentProcessor
from bm.datamanipulation.AdjustDataFrame import create_figure, import_mysql_table_csv
from bm.db_helper.AttributesHelper import get_features, get_labels, get_model_name
from bm.controllers.timeforecasting.TimeForecastingController import TimeForecastingController
from bm.utiles.CVSReader import getcvsheader
from bm.utiles.CVSReader import improve_data_file
from bm.controllers.prediction.ModelController import run_prediction_model, predict_values_from_model, get_model_status, \
    delet_model
from bm.apis.v1.APIsServices import predictvalues
from bm.datamanipulation.DataCoderProcessor import DataCoderProcessor

from app.base import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1000 * 1000
app.config['UPLOAD_FOLDER'] = 'app/data/'
app.config['DOCS_TEMPLATES_FOLDER'] = 'docs_templates/'
app.config['APPS_DATA_FOLDER'] = 'app/docs_templates/apps_data_sources'
app.config['DOWNLOAD_APPS_DATA_FOLDER'] = 'docs_templates/apps_data_sources'
app.config['OUTPUT_DOCS'] = 'app/base/output_docs/'
app.config['OUTPUT_PDF_DOCS'] = '/output_docs/'
app.config['DEMO_KEY'] = 'DEMO'
root_path = app.root_path


@blueprint.route('/index')
@login_required
def index():
    return render_template('base/index.html', segment='index')


@blueprint.route('/<template>')
@login_required
def route_template(template):
    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/base/FILE.html
        return render_template("base/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('base/page-404.html'), 404

    except:
        return render_template('base/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):
    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None


@blueprint.route('/selectmodelgoal')
@login_required
def selectmodelgoal():
    return render_template('applications/pages/selectmodelgoal.html', segment='selectmodelgoal')


@blueprint.route('/analysedata')
@login_required
def analysedata():
    return render_template('applications/pages/selectdssource.html', segment='selectmodelgoal')


@blueprint.route('/createmodel')
@login_required
def createmodel():
    ds_goal = request.args.get("t")
    return render_template('applications/pages/selectdssource.html', ds_goal=ds_goal, segment='createmodel')


@blueprint.route('/selectcsvds', methods=['GET', 'POST'])
@login_required
def selectcsvds():
    ds_source = request.form.get('ds_source')
    ds_goal = request.form.get("ds_goal")
    return selectds(ds_source, ds_goal)


@blueprint.route('/selectmysqlds', methods=['GET', 'POST'])
@login_required
def selectmysqlds():
    ds_source = request.form.get('ds_source')
    ds_goal = request.form.get("ds_goal")
    return selectds(ds_source, ds_goal)


@blueprint.route('/selectsfds', methods=['GET', 'POST'])
@login_required
def selectsfds():
    ds_source = request.form.get('ds_source')
    ds_goal = request.form.get("ds_goal")
    return selectds(ds_source, ds_goal)


@blueprint.route('/selectgsds', methods=['GET', 'POST'])
@login_required
def selectgsds():
    ds_source = request.form.get('ds_source')
    ds_goal = request.form.get("ds_goal")
    return selectds(ds_source, ds_goal)


def selectds(ds_source, ds_goal):
    return render_template('applications/pages/selectds.html', ds_id=ds_source, ds_goal=ds_goal, segment='createmodel')


@blueprint.route('/uploadcsvds', methods=['GET', 'POST'])
@login_required
def uploadcsvds():
    if request.method == 'POST':
        f = request.files['filename']
        ds_source = request.form.get('ds_source')
        ds_goal = request.form.get('ds_goal')
        filePath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(filePath)

        # Remove empty columns
        data = pd.read_csv(filePath, sep=',', encoding='latin1')
        data = data.dropna(axis=1, how='all')
        data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        data.to_csv(filePath, index=False)
        data = pd.read_csv(filePath)

        # Check if the dataset if engough
        count_row = data.shape[0]
        message = 'No'
        if (count_row > 50):
            # Get the DS file header
            headersArray = getcvsheader(filePath)
            fname = secure_filename(f.filename)
            if (ds_goal == current_app.config['PREDICTION_MODULE']):
                return render_template('applications/pages/prediction/selectfields.html', headersArray=headersArray,
                                       fname=fname,
                                       ds_source=ds_source, ds_goal=ds_goal,
                                       segment='createmodel', message=message)
            elif (ds_goal == current_app.config['FORECASTING_MODULE']):
                forecasting_columns, depended_columns, datetime_columns = TimeForecastingController.analyize_dataset(
                    data)
                message = (message if ((len(forecasting_columns) != 0) and (
                        len(datetime_columns) != 0) and (
                                               len(depended_columns) != 0)) else 'Your data file doesn not have one or more required fields to build the timeforecasting model. The file should have:<ul><li>One or more ctaegoires columns</li><li>One or more time series columns</li><li>One or more columns with numerical values.</li></ul><br/>Please check your file and upload it again.')
                return render_template('applications/pages/forecasting/dsfileanalysis.html', headersArray=headersArray,
                                       fname=fname,
                                       ds_source=ds_source, ds_goal=ds_goal,
                                       segment='createmodel', message=Markup(message),
                                       forecasting_columns=forecasting_columns,
                                       depended_columns=depended_columns,
                                       datetime_columns=datetime_columns)
            elif (ds_goal == current_app.config['ROBOTIC_MODULE']):
                return render_template('applications/pages/robotics/selectfields.html', headersArray=headersArray,
                                       fname=fname,
                                       ds_source=ds_source, ds_goal=ds_goal,
                                       segment='createmodel', message=message)
            else:  # ds_goal = '' means user can't decide
                document_processor = DocumentProcessor()
                columns_list, nan_cols, final_columns_list, final_total_rows, numric_columns, datetime_columns = document_processor.document_analyzer(
                    filePath)
                return render_template('applications/pages/analysisreport.html', headersArray=headersArray,
                                       columns_list=columns_list, nan_cols=nan_cols,
                                       final_columns_list=final_columns_list, final_total_rows=final_total_rows,
                                       numric_columns=numric_columns,
                                       datetime_columns=datetime_columns,
                                       fname=fname,
                                       ds_source=ds_source, ds_goal=ds_goal,
                                       segment='createmodel', message=message)
        else:
            message = 'Uploaded data document does not have enough data, the document must have minimum 50 records of data for accurate processing.'
            return render_template('applications/pages/dashboard.html',
                                   message=message,
                                   ds_source=ds_source, ds_goal=ds_goal,
                                   segment='createmodel')


@blueprint.route('/dffrommysqldb', methods=['GET', 'POST'])
@login_required
def dffromdb():
    try:
        if request.method == 'POST':
            host_name = request.form.get('host_name')
            username = request.form.get('username')
            password = request.form.get('password')
            database_name = request.form.get('database_name')
            table_name = request.form.get('table_name')
            ds_source = request.form.get('ds_source')
            ds_goal = request.form.get('ds_goal')
            csv_file_location = import_mysql_table_csv(host_name, username, password, database_name, table_name,
                                                       app.config['UPLOAD_FOLDER'])

            # Remove empty columns
            data = pd.read_csv(csv_file_location)
            data = data.dropna(axis=1, how='all')
            data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
            data.to_csv(csv_file_location, index=False)
            data = pd.read_csv(csv_file_location)

            # Check if the dataset if engough
            count_row = data.shape[0]
            if (count_row > 50):
                # Get the DS file header
                headersArray = getcvsheader(csv_file_location)
                fname = table_name + ".csv"
                message = 'No'

                if (ds_goal == current_app.config['PREDICTION_MODULE']):
                    return render_template('applications/pages/prediction/selectfields.html', headersArray=headersArray,
                                           fname=fname,
                                           ds_source=ds_source,
                                           segment='createmodel', message='No')
                elif (ds_goal == current_app.config['FORECASTING_MODULE']):
                    forecasting_columns, depended_columns, datetime_columns = TimeForecastingController.analyize_dataset(
                        data)
                    message = (message if ((len(forecasting_columns) != 0) and (
                            len(datetime_columns) != 0) and (
                                                   len(depended_columns) != 0)) else 'Your data file doesn not have one or more required fields to build the timeforecasting model. The file should have:<ul><li>One or more ctaegoires columns</li><li>One or more time series columns</li><li>One or more columns with numerical values.</li></ul><br/>Please check your file and upload it again.')
                    return render_template('applications/pages/forecasting/dsfileanalysis.html',
                                           headersArray=headersArray,
                                           fname=fname,
                                           ds_source=ds_source, ds_goal=ds_goal,
                                           segment='createmodel', message=Markup(message),
                                           forecasting_columns=forecasting_columns,
                                           depended_columns=depended_columns,
                                           datetime_columns=datetime_columns)
                elif (ds_goal == current_app.config['ROBOTIC_MODULE']):
                    return render_template('applications/dashboard.html')
                else:  # ds_goal = '' means user can't decide
                    return render_template('applications/dashboard.html')
            else:
                return render_template('applications/pages/dashboard.html',
                                       message='Uploaded data document does not have enough data, the document must have minimum 50 records of data for accurate processing.',
                                       ds_source=ds_source,
                                       segment='createmodel')
    except Exception as e:
        print(e)
    return render_template('page-500.html', error=e.with_traceback(), segment='error')


@blueprint.route('/creatingthemodel', methods=['GET', 'POST'])
@login_required
def creatingthemodel():
    try:
        if request.method == 'POST':
            fname = request.form.get('fname')
            ds_source = request.form.get('ds_source')
            ds_goal = request.form.get('ds_goal')
            loading_icon_path = os.path.join('images/', 'loading_icon.gif')
            progress_icon_path = os.path.join('images/', 'progress_icon_2.gif')
            if (ds_goal == current_app.config['PREDICTION_MODULE']):
                predictionvalues = numpy.array((request.form.getlist('predcitedvalues')))
                return render_template('applications/pages/prediction/creatingpredictionmodel.html',
                                       predictionvalues=predictionvalues,
                                       progress_icon_path=progress_icon_path, fname=fname,
                                       loading_icon_path=loading_icon_path,
                                       ds_source=ds_source, ds_goal=ds_goal,
                                       segment='createmodel')
            elif (ds_goal == current_app.config['FORECASTING_MODULE']):
                timefactor = request.form.get('timefactor')
                forecastingfactor = request.form.get('forecastingfactor')
                dependedfactor = request.form.get('dependedfactor')
                return render_template('applications/pages/forecasting/creatingforecastingmodel.html',
                                       timefactor=timefactor, dependedfactor=dependedfactor,
                                       forecastingfactor=forecastingfactor,
                                       progress_icon_path=progress_icon_path, fname=fname,
                                       loading_icon_path=loading_icon_path,
                                       ds_source=ds_source, ds_goal=ds_goal,
                                       segment='createmodel')
            else:
                return 0
        else:
            return 0
    except Exception as e:
        tb = sys.exc_info()[2]
        print(e)
        return render_template('page-500.html', error=e.with_traceback(tb))


@blueprint.route('/sendvalues', methods=['GET', 'POST'])
@login_required
def sendvalues():
    try:
        if request.method == 'POST':
            fname = request.form.get('fname')
            ds_source = request.form.get('ds_source')
            ds_goal = request.form.get('ds_goal')
            data_file_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            df = pd.read_csv(data_file_path, sep=",")
            data_sample = (df.sample(n=5))
            if (ds_goal == current_app.config['PREDICTION_MODULE']):
                predictionvalues = numpy.array((request.form.getlist('predcitedvalues')))
                idf = improve_data_file(fname, app.config['UPLOAD_FOLDER'], predictionvalues)
                # run model
                model_controller = run_prediction_model(root_path, data_file_path, predictionvalues, ds_source, ds_goal,
                                                        app.config['DEMO_KEY'])

                # Webpage details
                page_url = request.host_url + "predictevalues"
                page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"

                # APIs details and create APIs document
                model_api_details = ModelAPIDetails.query.first()
                apihelper = APIHelper()
                model_name = ModelProfile.query.with_entities(ModelProfile.model_name).first()[0]
                generate_apis_docs = apihelper.generateapisdocs(model_name,
                                                                str(request.host_url + 'api/' + model_api_details.api_version),
                                                                app.config['DOCS_TEMPLATES_FOLDER'],
                                                                app.config['OUTPUT_DOCS'])

                return render_template('applications/pages/prediction/modelstatus.html',
                                       accuracy=model_controller['accuracy'],
                                       confusion_matrix=model_controller['confusion_matrix'],
                                       plot_image_path=model_controller['plot_image_path'], sample_data=[
                        data_sample.to_html(border=0, classes='table table-hover', header="false",
                                            justify="center").replace(
                            "<th>", "<th class='text-warning'>")],
                                       Mean_Absolute_Error=model_controller['Mean_Absolute_Error'],
                                       Mean_Squared_Error=model_controller['Mean_Squared_Error'],
                                       Root_Mean_Squared_Error=model_controller['Root_Mean_Squared_Error'],
                                       segment='createmodel', page_url=page_url, page_embed=page_embed,
                                       created_on=model_controller['created_on'],
                                       updated_on=model_controller['updated_on'],
                                       last_run_time=model_controller['last_run_time'],
                                       fname=model_controller['file_name'])
            elif (ds_goal == current_app.config['FORECASTING_MODULE']):
                ml_forecasting_controller = MLForecastingController()
                forecastingfactor = request.form.get('forecastingfactor')
                dependedfactor = request.form.get('dependedfactor')
                timefactor = request.form.get('timefactor')
                all_return_values = ml_forecasting_controller.run_mlforecasting_model(
                    data_file_path, forecastingfactor,
                    dependedfactor, timefactor, ds_source, ds_goal)
                # Forecasting webpage details
                page_url = request.host_url + "embedforecasting"
                page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"

                # APIs details and create APIs document
                model_api_details = ModelAPIDetails.query.first()
                apihelper = APIHelper()
                model_name = ModelProfile.query.with_entities(ModelProfile.model_name).first()[0]
                generate_apis_docs = apihelper.generateapisdocs(model_name,
                                                                str(request.host_url + 'api/' + model_api_details.api_version),
                                                                app.config['DOCS_TEMPLATES_FOLDER'],
                                                                app.config['OUTPUT_DOCS'])

                return render_template('applications/pages/forecasting/modelstatus.html',
                                       depended_factor=dependedfactor, forecasting_category=forecastingfactor,
                                       plot_image_path=all_return_values['plot_image_path'],
                                       error_mse=all_return_values['error_mse'], sample_data=[
                        data_sample.to_html(border=0, classes='table table-hover', header="false",
                                            justify="center").replace("<th>", "<th class='text-warning'>")],
                                       fname=model_name,
                                       segment='createmodel', page_url=page_url, page_embed=page_embed,
                                       created_on=all_return_values['created_on'],
                                       updated_on=all_return_values['updated_on'],
                                       last_run_time=all_return_values['last_run_time']
                                       )
            else:
                return 0
    except Exception as e:
        tb = sys.exc_info()[2]
        profile = get_model_status()
        if ((len(profile) > 0) and ((profile['ds_goal'] == current_app.config['PREDICTION_MODULE']) or (
                profile['ds_goal'] == current_app.config['FORECASTING_MODULE']))):
            return redirect(url_for('base_blueprint.showdashboard'))
        else:
            return render_template('page-500.html',
                                   error="There is no enugh data to build the model after removing empty rows. The data set should have mimimum 50 records to buld the model.",
                                   segment='error')


@blueprint.route('/runthemodel_', methods=['GET', 'POST'])
@login_required
def runthemodel_():
    try:
        predictionvalues = numpy.array(ModelLabels.query.with_entities(ModelLabels.label_name).all()).flatten()
        fname = ModelProfile.query.with_entities(ModelProfile.model_name).first()[0] + '.csv'
        ds_source = ModelProfile.query.with_entities(ModelProfile.ds_source).first()[0]

        # run model
        data_file_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        df = pd.read_csv(data_file_path, sep=",")
        data_sample = (df.sample(n=5))
        model_controller = run_prediction_model(root_path, data_file_path, predictionvalues, ds_source,
                                                app.config['DEMO_KEY'])

        # Webpage details
        page_url = request.host_url + "predictevalues"
        page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"

        # APIs details
        # model_api_details = get_api_details()
        # APIs details and create APIs document
        model_api_details = ModelAPIDetails.query.first()
        apihelper = APIHelper()
        generate_apis_docs = apihelper.generateapisdocs(fname,
                                                        str(request.host_url + 'api/' + model_api_details.api_version),
                                                        app.config['DOCS_TEMPLATES_FOLDER'],
                                                        app.config['OUTPUT_DOCS'])

        return render_template('applications/pages/prediction/modelstatus.html', accuracy=model_controller['accuracy'],
                               confusion_matrix=model_controller['confusion_matrix'],
                               plot_image_path=model_controller['plot_image_path'], sample_data=[
                data_sample.to_html(border=0, classes='table table-hover', header="false",
                                    justify="center").replace(
                    "<th>", "<th class='text-warning'>")],
                               Mean_Absolute_Error=model_controller['Mean_Absolute_Error'],
                               Mean_Squared_Error=model_controller['Mean_Squared_Error'],
                               Root_Mean_Squared_Error=model_controller['Root_Mean_Squared_Error'],
                               segment='createmodel', page_url=page_url, page_embed=page_embed,
                               created_on=model_controller['created_on'], updated_on=model_controller['updated_on'],
                               last_run_time=model_controller['last_run_time'], fname=fname)
    except Exception as e:
        print(e)
        return render_template('page-500.html', error="blablabl", segment='error')


@blueprint.route('/runthemodel', methods=['GET', 'POST'])
@login_required
def runthemodel():
    try:
        ds_source = request.form.get('ds_source')
        return redirect('pages/selectds.html', ds_id=ds_source, segment='createmodel')

    except Exception as e:
        print(e)
        return render_template('page-500.html', error=e.with_traceback(), segment='error')


@app.route('/plot.png')
@login_required
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png', segment='createmodel')


@blueprint.route('/predictevalues', methods=['GET', 'POST'])
def predictevalues():
    try:
        features_list = get_features()
        if (len(features_list) > 0):
            labels_list = get_labels()
            testing_values = []
            opt_param = len(request.form)
            all_gategories_values = DataCoderProcessor.get_all_gategories_values()
            if opt_param == 0:
                # response = make_response()
                return render_template('applications/pages/prediction/predictevalues.html', features_list=features_list,
                                       labels_list=labels_list,
                                       predicted_value='nothing', testing_values='nothing',
                                       all_gategories_values=all_gategories_values, predicted='Nothing', message='No')
            else:
                if request.method == 'POST':
                    for i in features_list:
                        feature_value = request.form.get(i)
                        # final_feature_value = float(feature_value) if feature_value.isnumeric() else feature_value
                        final_feature_value = feature_value
                        testing_values.append(final_feature_value)
                    model_name = get_model_name()
                    predicted_value = predict_values_from_model(model_name, testing_values)
                    # response = make_response()
                    return render_template('applications/pages/prediction/predictevalues.html',
                                           features_list=features_list,
                                           labels_list=labels_list,
                                           predicted_value=predicted_value, testing_values=testing_values,
                                           all_gategories_values=all_gategories_values, predicted='Yes', message='No')
        else:
            return render_template('applications/pages/prediction/predictevalues.html',
                                   message='There is no active model')

    except Exception as e:
        etype, value, tb = sys.exc_info()
        print(traceback.print_exception(etype, value, tb))
        return render_template('page-500.html',
                               error='Not able to predict. One or more entered values has not relevant value in your dataset, please enter data from provided dataset',
                               segment='message')


@blueprint.route('/embedforecasting', methods=['GET', 'POST'])
def embedforecasting():
    try:
        profile = get_model_status()
        if len(profile) == 0:
            # response = make_response()
            return render_template('applications/pages/forecasting/embedforecasting.html',
                                   message='There is no active model')
        else:
            # Forecasting webpage details
            page_url = request.host_url + "embedforecasting"
            page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"
            return render_template('applications/pages/forecasting/embedforecasting.html',
                                   depended_factor=profile['depended_factor'],
                                   forecasting_factor=profile['forecasting_category'],
                                   plot_image_path=profile['plot_image_path'], message='No', )
    except Exception as e:
        print(e)
        return render_template('page-500.html',
                               error='Not able to predict. One or more entered values has not relevant value in your dataset, please enter data from provided dataset',
                               segment='message')


@blueprint.route('/showdashboard')
@login_required
def showdashboard():
    profile = get_model_status()

    if len(profile) > 0:
        fname = profile['model_name'] + '.csv'
        data_file_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        df = pd.read_csv(data_file_path, sep=",")
        data_sample = (df.sample(n=5, replace=True))

        if profile['ds_goal'] == current_app.config['PREDICTION_MODULE']:
            # Webpage details
            page_url = request.host_url + "predictevalues"
            page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"
            return render_template('applications/pages/prediction/dashboard.html',
                                   accuracy=profile['prediction_results_accuracy'],
                                   confusion_matrix='',
                                   plot_image_path=profile['plot_image_path'], sample_data=[
                    data_sample.to_html(border=0, classes='table table-hover', header="false",
                                        justify="center").replace(
                        "<th>", "<th class='text-warning'>")], Mean_Absolute_Error=profile['mean_absolute_error'],
                                   Mean_Squared_Error=profile['mean_squared_error'],
                                   Root_Mean_Squared_Error=profile['root_mean_squared_error'], message='No',
                                   fname=profile['model_name'], page_url=page_url, page_embed=page_embed,
                                   segment='showdashboard', created_on=profile['created_on'],
                                   updated_on=profile['updated_on'], last_run_time=profile['last_run_time'])
        elif profile['ds_goal'] == current_app.config['FORECASTING_MODULE']:
            # Forecasting webpage details
            page_url = request.host_url + "embedforecasting"
            page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"
            return render_template('applications/pages/forecasting/dashboard.html',
                                   accuracy=profile['prediction_results_accuracy'],
                                   confusion_matrix='', depended_factor=profile['depended_factor'],
                                   forecasting_factor=profile['forecasting_category'],
                                   error_mse=profile['mean_squared_error'],
                                   plot_image_path=profile['plot_image_path'], sample_data=[
                    data_sample.to_html(border=0, classes='table table-hover', header="false",
                                        justify="center").replace(
                        "<th>", "<th class='text-warning'>")], message='No',
                                   fname=profile['model_name'], page_url=page_url, page_embed=page_embed,
                                   segment='showdashboard', created_on=profile['created_on'],
                                   updated_on=profile['updated_on'], last_run_time=profile['last_run_time'])
        else:
            return 0
    else:
        return render_template('applications/pages/dashboard.html', message='You do not have any running model yet.',
                               segment='showdashboard')


@blueprint.route('/deletemodel', methods=['GET', 'POST'])
@login_required
def deletemodel():
    delete_model = delet_model()
    return render_template('applications/pages/dashboard.html', message='You do not have any running model yet.',
                           segment='deletemodel')


@blueprint.route('/applications', methods=['GET', 'POST'])
@login_required
def applications():
    return render_template('applications/applications.html', segment='applications')


@blueprint.route('/installapp', methods=['GET', 'POST'])
@login_required
def installapp():
    if request.method == 'POST':
        delete_model = delet_model()
        f = request.form.get('filename')
        original = os.path.join(app.config['APPS_DATA_FOLDER'], f)
        target = os.path.join(app.config['UPLOAD_FOLDER'], f)

        shutil.copyfile(original, target)

        # Remove empty columns
        data = pd.read_csv(target)
        data = data.dropna(axis=1, how='all')
        data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        data.to_csv(target, index=False)
        data = pd.read_csv(target)

        # Check if the dataset if engough
        count_row = data.shape[0]
        if (count_row > 50):
            # Get the DS file header
            headersArray = getcvsheader(target)
            fname = f
            return render_template('applications/pages/prediction/selectfields.html', headersArray=headersArray,
                                   fname=fname,
                                   segment='createmodel', message='No')
        else:
            return render_template('applications/pages/prediction/selectfields.html',
                                   message='Uploaded data document does not have enough data, the document must have minimum 50 records of data for accurate processing.',
                                   segment='createmodel')


@blueprint.route('/downloaddsfile', methods=['GET', 'POST'])
@login_required
def downloaddsfile():
    if request.method == 'POST':
        f = request.form.get('filename')
        path = os.path.join(app.config['DOWNLOAD_APPS_DATA_FOLDER'], f)
        return send_file(path, as_attachment=True)


## APIs

@blueprint.route('/api/v1/predictevalues', methods=['POST'])
def predictevalues_api():
    content = request.json
    # apihelper = APIHelper()
    # apireturn_json = apihelper.api_runner(content)  # predictvalues(content['inputs'])
    # inputs = content['inputs']
    apireturn_json = predictvalues(content)
    return apireturn_json


@blueprint.route('/downloadapisdocument', methods=['GET', 'POST'])
@login_required
def downloadapisdocument():
    # For windows you need to use drive name [ex: F:/Example.pdf]
    fname = ModelProfile.query.with_entities(ModelProfile.model_name).first()[0]
    path = root_path + app.config['OUTPUT_PDF_DOCS'] + '/' + fname + '_BrontoMind_APIs_document.docx'
    return send_file(path, as_attachment=True)


## Errors

@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('page-403.html', segment='error'), 403


@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('page-403.html', segment='error'), 403


@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('page-404.html', segment='error'), 404


@blueprint.errorhandler(500)
def internal_error(error):
    return render_template(('page-500.html'), error=error, segment='error'), 500


@blueprint.errorhandler(413)
def request_entity_too_large_error(error):
    return render_template(('page-413.html'), segment='error'), 413
