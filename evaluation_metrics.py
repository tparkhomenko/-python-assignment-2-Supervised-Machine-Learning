import numpy as np


class EvaluationMetrics:
    _mock_result = np.array([0, 0])

    # TODO: code __init__
    def __init__(self, y_true, y_pred):
        self._y_true = y_true
        self._y_pred = y_pred
        self._uniq = EvaluationMetrics._unique_values(y_true, y_pred)
        self._matrix = self.confusion_matrix
        self._diag_sum = np.sum(np.diag(self._matrix))

    @staticmethod
    def _unique_values(y_true, y_pred):
        return sorted(list(set(np.concatenate((y_true, y_pred)))))

    @property
    def confusion_matrix(self):
        y_true, y_pred = self._y_true, self._y_pred
        matrix = np.zeros((len(self._uniq), len(self._uniq)), dtype=int)
        for i in range(0, len(y_true)):
            matrix[y_true[i], y_pred[i]] += 1
        return matrix

    # noinspection PyPep8Naming
    @property
    def FP(self):
        return self._mock_result

    # noinspection PyPep8Naming
    @property
    def FP(self):
        sum_rows = []
        # test[1,:]
        for class_i in self._uniq:
            sum_rows.append(np.sum(self._matrix[:, class_i]) - self.TP[class_i])
        return sum_rows

    # noinspection PyPep8Naming
    @property
    def TP(self):
        return [self._matrix[class_i, class_i] for class_i in self._uniq]

    # noinspection PyPep8Naming
    @property
    def TN(self):
        sum_tn = []
        for class_i in self._uniq:
            sum_tn.append(np.sum(self._matrix) - self.TP[class_i] - self.FP[class_i] - self.FN[class_i])
        return sum_tn

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

    @property
    def accuracy_score(self):
        return float(np.sum(self.TP)) / np.sum(self._matrix)

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
