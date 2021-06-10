import numpy as np


class EvaluationMetrics:
    #_mock_result = np.array([0, 0])

    def __init__(self, y_true, y_pred):
        self._y_true = y_true
        self._y_pred = y_pred
        self._uniq = EvaluationMetrics._unique_values(y_true, y_pred)
        self._matrix = self.confusion_matrix
        self._diag_sum = np.sum(np.diag(self._matrix))

    @staticmethod
    def _unique_values(y_true, y_pred):  # finds unique classes, mekes set from then, gives a list back
        return sorted(list(set(np.concatenate((y_true, y_pred)))))

    @property
    def confusion_matrix(self):  # заполняет confusion матрицу
        y_true, y_pred = self._y_true, self._y_pred
        matrix = np.zeros((len(self._uniq), len(self._uniq)), dtype=int)
        for i in range(0, len(y_true)):
            matrix[y_true[i], y_pred[i]] += 1
        return matrix

    # noinspection PyPep8Naming
    @property
    def FN(self):
        sum_columns = []
        # test[1,:]
        for class_i in self._uniq:
            sum_columns.append(np.sum(self._matrix[class_i, :]) - self.TP[class_i])
        return np.array(sum_columns)

    # noinspection PyPep8Naming
    @property
    def FP(self):
        sum_rows = []
        # test[1,:]
        for class_i in self._uniq:
            sum_rows.append(np.sum(self._matrix[:, class_i]) - self.TP[class_i])
        return np.array(sum_rows)

    # noinspection PyPep8Naming
    @property
    def TP(self):
        return np.array([self._matrix[class_i, class_i] for class_i in self._uniq])

    # noinspection PyPep8Naming
    @property
    def TN(self):
        sum_tn = []
        for class_i in self._uniq:
            sum_tn.append(np.sum(self._matrix) - self.TP[class_i] - self.FP[class_i] - self.FN[class_i])
        return np.array(sum_tn)

    @property
    def precision(self):
        return self.TP / (self.TP + self.FP)

    @property
    def recall(self):
        return self.TP / (self.TP + self.FN)

    # noinspection PyPep8Naming
    @property
    def F1(self):  # need to be optimized
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def accuracy_score(self):  # multiple calculations (-> save variables in init)
        return float(np.sum(self.TP)) / np.sum(self._matrix)

    @staticmethod
    def _get_numpy_str(np_list):
        size = len(np_list)
        return "\t\t\t".join([f"Class {i}: {str(round(np_list[i], 2))}" for i in range(size)])

    def __str__(self):
        return f"""Evaluation Summary:
\tAccuracy: {round(self.accuracy_score, 2)}
\tPrecision:
\t\t{EvaluationMetrics._get_numpy_str(self.precision)}			
\tRecall:
\t\t{EvaluationMetrics._get_numpy_str(self.recall)}			
\tF1 Score:
\t\t{EvaluationMetrics._get_numpy_str(self.F1)}				
\tTrue Positives:
\t\t{EvaluationMetrics._get_numpy_str(self.TP)}				
\tTrue Negatives:
\t\t{EvaluationMetrics._get_numpy_str(self.TN)}				
\tFalse Positives:
\t\t{EvaluationMetrics._get_numpy_str(self.FP)}		
\tFalse Negatives:
\t\t{EvaluationMetrics._get_numpy_str(self.FN)}	
"""
