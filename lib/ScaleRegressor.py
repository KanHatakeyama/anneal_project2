from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import numpy as np
#from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression


class ScaleRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper class of sklearn. X and y will be standardized automatically

    Attributes
    ----------
    model : object
        sklearn regressor object
    auto_scaling_X: bool
        if true, X will be scaled automatically
    auto_scaling_y: bool
        if true, y will be scaled automatically
    """

    def __init__(self,
                 model=PLSRegression(n_components=30),
                 auto_scaling_X=True,
                 auto_scaling_y=True,

                 ):
        self.model = model
        self.auto_scaling_X = auto_scaling_X
        self.auto_scaling_y = auto_scaling_y

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X, y):
        if self.auto_scaling_X:
            X = self.scaler_X.fit_transform(np.array(X))

        if self.auto_scaling_y:
            y = self.scaler_y.fit_transform(
                np.array(y).reshape(-1, 1)).reshape(-1)

        self.model.fit(X, y)
        try:
            self.coef_ = self.model.coef_
        except:
            pass
        return self

    def predict(self, X):
        pred_y = self._predict(X)
        return pred_y

    def _predict(self, X):
        if self.auto_scaling_X:
            X = self.scaler_X.transform(np.array(X))

        pred_y = self.model.predict(X)
        if self.auto_scaling_y:
            return self.scaler_y.inverse_transform(pred_y)
        else:
            return pred_y
