from sklearn.linear_model import LogisticRegression

class ModelEvaluator:
    """
    Used to evaluate weights on test dataset. Evaluation is performed with sklearn due to difficulties of
    changing weights for pyspark's logistic regression.
    """

    def __init__(self, X_test, Y_test):
        """
        Creates a logistic regression object whose weights will be overriden.
        :param X_test: numpy array of test inputs
        :param Y_test: numpy array of test labels
        """
        self.X_test = X_test
        self.Y_test = Y_test
        self.logisticRegr = LogisticRegression()
        self.logisticRegr.fit(self.X_test, self.Y_test)

    def accuracy(self, weights, intercepts):
        """
        Calculates accuracy on test dataset given new weights and intercepts
        :param weights: numpy array of weights
        :param intercepts: numpy array of intercepts
        :return: returns accuracy on test dataset
        """
        self.logisticRegr.coef_ = weights  # override weights and coefficients
        self.logisticRegr.intercept_ = intercepts
        return self.logisticRegr.score(self.X_test, self.Y_test)