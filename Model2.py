from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from typing import NoReturn
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

class Estimator2:
    """
    Polynomial Fitting using Least Squares estimation
    """

    def __init__(self):
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        super().__init__()
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.model.fit(X, y, sample_weight=None)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.model.predict(X)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return 1-self.model.score(X, y, sample_weight=None)
