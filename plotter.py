################################################################################
# Author:      Taisiya Parkhomenko
# MatNr:       01650051
# File:        plotter.py
# Description: Plotter.
# Comments:    -
################################################################################
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Plotter(ABC):
    _caption = "abstract"

    @staticmethod
    def get_plotter_classes():
        result = {}

        for subclass in Plotter.__subclasses__():
            result[subclass._caption] = subclass
        return result

    @staticmethod
    @abstractmethod
    def plot_feature(feature_name, feature_values):
        pass

    @staticmethod
    @abstractmethod
    def plot_feature_vs_target(feature_name, feature_values, target_name, target_values):
        pass

    @staticmethod
    def _get_plot_dataframe(feature_name, feature_values):
        return pd.DataFrame(feature_values, columns=[feature_name])

    @staticmethod
    def _get_multiple_dataset(feature_name, feature_values, target_name, target_values):
        cols = {feature_name: feature_values, target_name: target_values}
        df = pd.DataFrame(data=cols)
        unique_target = np.unique(target_values)
        return df, unique_target

    @staticmethod
    def _split_by_target(feature_name, feature_values, target_name, target_values):
        df, unique_target = Plotter._get_multiple_dataset(feature_name, feature_values, target_name, target_values)
        values_by_target = []
        labels_by_target = []
        for target_value in unique_target:
            values_by_target.append(df[df[target_name] == target_value][feature_name].tolist())
            labels_by_target.append(str(target_value))
        return values_by_target, labels_by_target


class BarPlotter(Plotter):
    _caption = "bar"

    @staticmethod
    def plot_feature(feature_name, feature_values):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.bar(np.arange(1, len(feature_values) + 1), feature_values)
        ax.set_ylabel(feature_name)
        fig.suptitle(feature_name)
        return fig

    @staticmethod
    def plot_feature_vs_target(feature_name, feature_values, target_name, target_values):
        df, unique_target = BarPlotter._get_multiple_dataset(feature_name, feature_values, target_name, target_values)

        fig, ax = plt.subplots(nrows=1, ncols=len(unique_target))
        for target_value_i in range(0, len(unique_target)):
            target_value = unique_target[target_value_i]
            filtered_values = df[df[target_name] == target_value][feature_name].tolist()
            ax[target_value_i].bar(np.arange(1, len(filtered_values) + 1), filtered_values)
            ax[target_value_i].set_xlabel(f"{target_name}: {target_value}")
            ax[target_value_i].set_ylabel(feature_name)

        fig.suptitle(f"{feature_name} by {target_name}")
        return fig


class HistogramPlotter(Plotter):
    _caption = "histogram"

    @staticmethod
    def plot_feature(feature_name, feature_values):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.hist(feature_values, 10, density=True, histtype='bar')
        fig.suptitle(feature_name)
        return fig

    @staticmethod
    def plot_feature_vs_target(feature_name, feature_values, target_name, target_values):
        values_by_target, labels_by_target = HistogramPlotter._split_by_target(feature_name, feature_values,
                                                                               target_name, target_values)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.hist(values_by_target, 10, density=True, histtype='bar', label=labels_by_target)

        fig.suptitle(f"{feature_name} by {target_name}")
        return fig


class BoxPlotter(Plotter):
    _caption = "box"

    @staticmethod
    def plot_feature(feature_name, feature_values):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.boxplot(feature_values)
        fig.suptitle(feature_name)
        return fig

    @staticmethod
    def plot_feature_vs_target(feature_name, feature_values, target_name, target_values):
        values_by_target, labels_by_target = BoxPlotter._split_by_target(feature_name, feature_values, target_name,
                                                                         target_values)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.boxplot(values_by_target, labels=labels_by_target)

        fig.suptitle(f"{feature_name} by {target_name}")
        return fig
