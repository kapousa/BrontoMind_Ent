import json

from bm.apis.v1.APIsServices import NpEncoder
from bm.db_helper.AttributesHelper import get_model_name, get_features, get_labels


class APISclassificationServices:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    def classify_data(self, content):
        model_name = get_model_name()
        features_list = get_features()
        lables_list = get_labels()
        testing_values = []
        for i in features_list:
            feature_value = str(content[i])
            final_feature_value = feature_value  # float(feature_value) if feature_value.isnumeric() else feature_value
            testing_values.append(final_feature_value)
        predicted_value = '' #predict_values_from_model(model_name, testing_values)

        # Create predicted values json object
        predicted_values_json = {}
        for j in range(len(predicted_value)):
            for i in range(len(lables_list)):
                bb = predicted_value[j][i]
                predicted_values_json[lables_list[i]] = predicted_value[j][i]
                # NpEncoder = NpEncoder(json.JSONEncoder)
            json_data = json.dumps(predicted_values_json, cls=NpEncoder)

        return json_data
