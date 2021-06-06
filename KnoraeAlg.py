import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class KnoraeAlg(BaseEnsemble,ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=5, hard_voting=True, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.hard_voting = hard_voting
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self,X,y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)


    def predict(self,X):
        check_is_fitted(self, "classes_")