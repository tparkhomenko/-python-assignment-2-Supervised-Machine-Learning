################################################################################
# Author:      Taisiya Parkhomenko
# MatNr:       01650051
# File:        dataset_explorer.py
# Description: Plotting features and calculating simple metrics.
# Comments:    -
################################################################################
import plotter
import math
import numpy as np
from scipy import stats


class DatasetExplorer:

    def __init__(self, dataset, plot_types):
        self._dataset = dataset
        self._plot_types = plot_types

    def __str__(self):
        return f'DatasetExplorer Info:\n\tDataset name: {self._dataset.name}'

    def plot_feature(self, feature_name):
        plot_type = self._plot_types[feature_name]

        feature_list = self._dataset.get_feature_values(feature_name)

        if plot_type == "bar":
            return plotter.BarPlotter.plot_feature(feature_name, feature_list)
        elif "hist" in plot_type:
            return plotter.HistogramPlotter.plot_feature(feature_name, feature_list)
        elif "box" in plot_type:
            return plotter.BoxPlotter.plot_feature(feature_name, feature_list)
        else:
            raise ValueError('Specify plottype: bar/hist/box')

    def plot_feature_vs_target(self, feature_name):
        plot_type = self._plot_types[feature_name]

        target_name = self._dataset.target_feature_name

        feature_list = self._dataset.get_feature_values(feature_name)
        target_list = self._dataset.get_feature_values(target_name)

        if plot_type == "bar":
            return plotter.BarPlotter.plot_feature_vs_target(feature_name, feature_list, target_name, target_list)
        elif "hist" in plot_type:
            return plotter.HistogramPlotter.plot_feature_vs_target(feature_name, feature_list, target_name, target_list)
        elif "box" in plot_type:
            return plotter.BoxPlotter.plot_feature_vs_target(feature_name, feature_list, target_name, target_list)
        else:
            raise ValueError('Specify plottype: bar/hist/box')

    @staticmethod
    def _calculate_p_value(points_number, r):
        freedom_degree = points_number - 2
        t_value = r * math.sqrt(freedom_degree) / math.sqrt(1 - r * r)
        s_value = np.random.standard_t(freedom_degree, size=1000000)
        pre_p = np.sum(s_value < t_value) / float(len(s_value))
        return 2 * min(pre_p, 1 - pre_p)

    @staticmethod
    def _rank_data(a):
        arr = np.ravel(np.asarray(a))
        sorter = np.argsort(arr, kind='quicksort')

        inv = np.empty(sorter.size, dtype=np.intp)
        inv[sorter] = np.arange(sorter.size, dtype=np.intp)

        arr = arr[sorter]
        obs = np.r_[True, arr[1:] != arr[:-1]]
        dense = obs.cumsum()[inv]

        # cumulative counts of each unique value
        count = np.r_[np.nonzero(obs)[0], len(obs)]

        # average method
        return .5 * (count[dense] + count[dense - 1] + 1)

    def calculate_correlation_between_features(
            self,
            feature_name_1,
            feature_name_2,
            method="pearson"
    ):

        features = self._dataset.get_feature_values(feature_name_1, feature_name_2, remove_missing=True)

        if method == "pearson":
            r = np.corrcoef(features)[0, 1]
            p = DatasetExplorer._calculate_p_value(len(features[0]), r)
            # corr = stats.pearsonr(features[0], features[1])  # scipy version
            return r, p
        elif method == "spearman":
            ranked_features_1 = DatasetExplorer._rank_data(features[0])
            ranked_features_2 = DatasetExplorer._rank_data(features[1])
            r = np.corrcoef(ranked_features_1, ranked_features_2)[0, 1]
            p = DatasetExplorer._calculate_p_value(len(features[0]), r)
            # corr = stats.spearmanr(features[0], features[1])  # scipy version
            return r, p
        elif method == "kendalltau":
            corr = stats.kendalltau(features[0], features[1])
            return corr[0], corr[1]

    def print_quartiles_of_feature(self, feature_name):
        feature_list = self._dataset.get_feature_values(feature_name, remove_missing=True)

        q1 = np.percentile(feature_list, 25)  # Q1
        q2 = np.percentile(feature_list, 50)  # Q2 median
        q3 = np.percentile(feature_list, 75)  # Q3

        output = f"""
Quartiles:
    Q1: {q1}
    Q2: {q2}
    Q3: {q3}
"""
        print(output)

    def print_feature_names(self):
        print("Feature names:\n" + ", ".join(self._dataset.feature_names))

    def print_feature_description(self, feature_name=None):
        if feature_name is None:
            print('Features and descriptions:')
            for feature_key in self._dataset.column_info:
                print(f'\t{feature_key}: {self._dataset.column_info[feature_key]}')
        else:
            print(f'{feature_name}: {self._dataset.column_info[feature_name]}')
