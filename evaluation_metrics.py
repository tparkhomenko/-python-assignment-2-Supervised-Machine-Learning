import numpy as np


class EvaluationMetrics:
    _mock_result = np.array([0, 0])

    # TODO: code __init__
    def __init__(self, y_true, y_pred):
        self._y_true = y_true
        self._y_pred = y_pred
        self._matrix = self.confusion_matrix

    @staticmethod
    def _unique_values(y_true, y_pred):
        return sorted(list(set(np.concatenate((y_true, y_pred)))))

    @property
    def confusion_matrix(self):
        y_true, y_pred = self._y_true, self._y_pred
        uniq = EvaluationMetrics._unique_values(y_true, y_pred)
        matrix = np.zeros((len(uniq), len(uniq)), dtype=int)
        for i in range(0, len(y_true)):
            matrix[y_pred[i], y_true[i]] += 1
        return matrix

    # TODO: code getter FP
    # noinspection PyPep8Naming
    @property
    def FP(self):
        return self._mock_result

    # TODO: code getter FN
    # noinspection PyPep8Naming
    @property
    def FN(self):
        return self._mock_result

    # noinspection PyPep8Naming
    @property
    def TP(self):
        return np.diag(self._matrix)

    # TODO: code getter TN
    # noinspection PyPep8Naming
    @property
    def TN(self):
        return self._mock_result

    # TODO: code getter precision
    @property
    def precision(self):
        return self._mock_result

    # TODO: code getter recall
    @property
    def recall(self):
        return self._mock_result

    # TODO: code getter F1
    # noinspection PyPep8Naming
    @property
    def F1(self):
        return self._mock_result

    # TODO: code getter accuracy_score
    @property
    def accuracy_score(self):
        return 0.99

    @staticmethod
    def _get_numpy_str(np_list):
        result_str = ""
        size = len(np_list)
        for i in range(size):
            temp_str = f"Class {i}: {np_list[i]}"
            result_str += temp_str
            if i != size - 1:
                result_str += " " * (30 - len(temp_str))
        return result_str

    def __str__(self):
        return f"""
Evaluation Summary:
    Accuracy: {self.accuracy_score}
    Precision:
        {EvaluationMetrics._get_numpy_str(self.precision)}			
    Recall:
        {EvaluationMetrics._get_numpy_str(self.recall)}			
    F1 Score:
        {EvaluationMetrics._get_numpy_str(self.F1)}				
    True Positives:
        {EvaluationMetrics._get_numpy_str(self.TP)}				
    True Negatives:
        {EvaluationMetrics._get_numpy_str(self.TN)}				
    False Positives:
        {EvaluationMetrics._get_numpy_str(self.FP)}		
    False Negatives:
        {EvaluationMetrics._get_numpy_str(self.FN)}	
"""
