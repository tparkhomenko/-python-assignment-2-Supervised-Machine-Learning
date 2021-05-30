import numpy as np

from evaluation_metrics import EvaluationMetrics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class DataPredictor:
    _best = 0

    def __init__(self, dataset, config_info_dict):
        self._dataset = dataset
        self._config_info_dict = config_info_dict
        self._features_train, self._target_train, self._features_test, self._target_test \
            = dataset.split_data(impute_strategy=config_info_dict['impute_strategy'])

        self._classifier_results = []

    def best_classifier(self, classifiers):
        classifier_results = []
        for classifier in classifiers:
            classifier.fit(self._features_train, self._target_train)
            target_predicted = classifier.predict(self._features_test)
            metrics = EvaluationMetrics(self._target_test, target_predicted)
            classifier_results.append({
                'fitted': classifier, 'target_predicted': target_predicted, 'metrics': metrics
            })

        classifier_results.sort(key=lambda result: result['metrics'].accuracy_score, reverse=True)
        self._classifier_results = classifier_results
        return classifier_results[self._best]['fitted']

    def print_evaluation_results(self):
        print('Classifiers Ranked by Accuracy:')
        for result in self._classifier_results:
            print('\t' + str(result['fitted']) + ' Accuracy Score: ' + str(result['metrics'].accuracy_score))
        print('Detailed info on the best classifier' + ' ' + str(
            self._classifier_results[self._best]['fitted']) + ': ' + '\n'
              + str(self._classifier_results[self._best]['metrics']))

    @staticmethod
    def _unique_values(y_true, y_pred):
        return sorted(list(set(np.concatenate((y_true, y_pred)))))

    @staticmethod
    def _get_classifier_name(classifier):
        return type(classifier).__name__

    @classmethod
    def _draw_confusion_matrix(cls, true_target, classifier_result, ax):
        matrix = classifier_result['metrics'].confusion_matrix
        display_labels = cls._unique_values(true_target, classifier_result['target_predicted'])
        drawing = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=display_labels)
        return drawing.plot(ax=ax)

    @classmethod
    def _draw_accuracies(cls, results, file_name):
        accuracies = []
        classifiers = []
        for result in results:
            accuracies.append(result['metrics'].accuracy_score)
            classifiers.append(str(result['fitted']))

        fig, ax = plt.subplots()
        fig.set_size_inches(4 * len(classifiers), 3)
        ax.bar(classifiers, accuracies)
        plt.savefig(file_name)

    @classmethod
    def _draw_chart_by_result(cls, true_target, results, drawer, file_path):
        classifiers_len = len(results)

        fig, axes = plt.subplots(nrows=1, ncols=classifiers_len)
        fig.set_size_inches(4 * classifiers_len, 3)
        for i in range(classifiers_len):
            result = results[i]
            drawer(true_target, result, axes[i])
            axes[i].set_title(str(result['fitted']))
            fig.tight_layout()
        plt.savefig(file_path)

    def visualize_classification_results(self):
        path = self._config_info_dict['classification_output_folder']
        self._draw_chart_by_result(self._target_test, self._classifier_results, self._draw_confusion_matrix,
                                   path + '/confusion_matrices.png')
        self._draw_accuracies(self._classifier_results, path + '/accuracy_comparison.png')
