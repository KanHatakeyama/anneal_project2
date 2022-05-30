import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from Fingerprint import Fingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

FP = Fingerprint()


class MoleculeRegressor(BaseEstimator, RegressorMixin):

    """
    wrapper class of rdkit and sklearn. it can input smiles string as X, converting to fingerprint

    Attributes
    ----------
    model : object
        sklearn regressor
    FP: object
        class to calculate fingerprint
    auto_scaling: bool
        if true, scale y automatically
    """

    def __init__(self,
                 model=SGDRegressor(),
                 FP=FP,
                 auto_scaling=True
                 ):
        self.model = model
        self.FP = FP
        self.auto_scaling = auto_scaling
        self.scaler = StandardScaler()

    def _calc_fp(self, X):
        fp, _ = self.FP.calc_fingerprint(X)
        return np.array(fp)

    def fit(self, X, y):
        # scale y
        if self.auto_scaling:
            y = self.scaler.fit_transform(y.reshape(-1, 1))

        # calc fingerprint
        self.model.fit(self._calc_fp(X), y)

        self.coef_ = self.model.coef_
        return self

    def predict(self, X):
        fp = self._calc_fp(X)
        pred_y = self._predict(fp)
        return pred_y

    def _predict(self, fp):
        pred_y = self.model.predict(fp)
        if self.auto_scaling:
            return self.scaler.inverse_transform(pred_y)
        else:
            return pred_y
