################################################################################
# Author:      Taisiya Parkhomenko
# MatNr:       01650051
# File:        dataset.py
# Description: Loading/writing data files and other dataset-related functionality.
# Comments:    -
################################################################################
import csv
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import json

from os import listdir
from os.path import isfile, join


class Dataset:

    @staticmethod
    def _read_json_column_info(
            config_info_dict):  # static method. It reads the config file and returns the content as a dict
        file_path = join(
            config_info_dict["data_folder"],
            config_info_dict["column_info_file"],
        )
        with open(file_path) as a:
            config_info_dict = json.load(a)
            return config_info_dict

    @staticmethod
    def _get_file_path(config_info_dict, file_name):
        return join(config_info_dict['data_folder'], file_name)

    @staticmethod
    def _get_csv_files(config_info_dict):
        data_folder = config_info_dict['data_folder']
        data_files = []
        for data_file in listdir(data_folder):
            file_path = Dataset._get_file_path(config_info_dict, data_file)
            if isfile(file_path) and data_file.endswith('.csv'):
                data_files.append(file_path)
        return data_files

    @staticmethod
    def _read_data(config_info_dict, feature_names):
        file_paths = Dataset._get_csv_files(config_info_dict)

        result = []  # Read files and combine to one dataset
        for file_path in file_paths:
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                csv_list = list(csv_reader)
                result = result + csv_list

        df = pd.DataFrame(result, columns=feature_names)

        na = config_info_dict['na_characters']  # na characters are replaced with None, other will be parsed to float
        for feature_name in feature_names:
            df[feature_name] = df[feature_name].map(lambda x: None if x in na else int(x) if x.isdigit() else float(x))

        if 'binarize_threshold' in config_info_dict:
            binarize_threshold = config_info_dict['binarize_threshold']
            target_name = config_info_dict['target_feature']
            df[target_name] = df[target_name].map(lambda x: 0 if x < binarize_threshold else 1)

        return df

    def __init__(self, config_info_dict):
        self._name = config_info_dict['dataset_name']
        self._column_info = Dataset._read_json_column_info(config_info_dict)
        self._target_feature_name = config_info_dict['target_feature']
        self._feature_names = list(config_info_dict['plot_types'].keys())
        self._na_characters = config_info_dict['na_characters']
        self._random_state_for_split = config_info_dict['random_state_for_split']
        self._test_size = config_info_dict['test_size']  # 0..1
        self._features_to_use_for_classification = config_info_dict['features_to_use_for_classification']

        self._data = Dataset._read_data(config_info_dict, self.feature_names)

    # getter/setter
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def target_feature_name(self):
        return self._target_feature_name

    @property
    def column_info(self):
        return self._column_info

    @property
    def feature_names(self):
        return self._feature_names

    def __str__(self):
        return f'Dataset Info: \n\t' \
               f'Dataset name: {self.name}\n\t' \
               f'Number of features: {len(self.feature_names)}, number of rows: {len(self._data.index)}'

    def write_data_to_file(self, output_file):  # writes the data to a .csv file (without column headers/names)
        self._data.to_csv(output_file, index=False, index_label=False, na_rep=self._na_characters)

    def get_feature_values(self, *feature_names,
                           remove_missing=False):  # Returns the values from the requested feature(s)
        result = self._data[list(set(feature_names))]

        if remove_missing:
            result = result.dropna()

        if len(feature_names) > 1:
            return [result[feature_name].tolist() for feature_name in feature_names]  # list of lists
        else:
            return result[feature_names[0]].tolist()  # list

    def get_target_values(self):  # Returns the values from all rows of the target column as a list.
        return self.get_feature_values(self.target_feature_name)

    @staticmethod
    def _get_feature_names_without_target(feature_names, target_name):
        return list(filter(lambda name: name != target_name, feature_names))

    def split_data(self, impute_strategy=None):
        feature_names = self._feature_names if self._features_to_use_for_classification[0] == "all" else self._features_to_use_for_classification

        validated_feature_names = self._get_feature_names_without_target(feature_names, self._target_feature_name)
        validated_feature_names.append(self._target_feature_name)

        values = np.array(self.get_feature_values(*validated_feature_names, remove_missing=impute_strategy is None))
        if impute_strategy is not None:
            imp_mean = SimpleImputer(strategy=impute_strategy)
            values = imp_mean.fit_transform(values)

        feature_values = values[:-1].transpose()
        target_values = values[-1]

        features_train, features_test, target_train, target_test = train_test_split(
            feature_values,
            target_values,
            test_size=self._test_size,
            random_state=self._random_state_for_split
        )

        return features_train, target_train, features_test, target_test
