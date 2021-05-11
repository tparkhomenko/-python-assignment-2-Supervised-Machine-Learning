from evaluation_metrics import EvaluationMetrics
from sklearn.metrics import plot_confusion_matrix


class DataPredictor:

    def __init__(self, dataset, config_info_dict):
        self._dataset = dataset
        self._config_info_dict = config_info_dict
        self._features_train, self._target_train, self._features_test, self._target_test \
            = dataset.split_data(impute_strategy=config_info_dict['impute_strategy'])

        self._classifier_results = []

    def best_classifier(self, classifiers):
        classifier_results = []
        for classifier in classifiers:
            fitted = classifier.fit(self._features_train, self._target_train)
            target_predicted = fitted.predict(self._features_test)
            metrics = EvaluationMetrics(self._target_test, target_predicted)
            classifier_results.append({
                'fitted': fitted, 'target_predicted': target_predicted, 'metrics': metrics
            })

        classifier_results.sort(key=lambda result: result['metrics'].accuracy_score)
        self._classifier_results = classifier_results
        return classifier_results[-1]['fitted']

    def print_evaluation_results(self):
        print(self._classifier_results[-1]['metrics'])

    # TODO: code visualize_classification_results
    def visualize_classification_results(self):
        plot_confusion_matrix(self._classifier_results[-1]['fitted'], self._features_test, self._target_test)
