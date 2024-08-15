from imbens.ensemble import SelfPacedEnsembleClassifier
import numpy as np
from balancingUnderSampler import BalancingSampler
from imbens.ensemble.base import BaseImbalancedEnsemble
from imbens.ensemble import EasyEnsembleClassifier
from collections import Counter

class BalancedEnsembleClassifier(BaseImbalancedEnsemble):
    def __init__(self,
                 estimator=None,
                 *,
                 n_estimators=50,
                 estimator_params=tuple(),
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.name = "Balanced_Ensemble"
        self.sampler_ = BalancingSampler()
        self.y_pred_proba = None
        self.y_pred_proba_list = []



    def fit(self, X, y, **kwargs):
        return super().fit(X, y, **kwargs)

    def _fit(self,
            X,
            y,
            **kwargs):

        n_estimators = self.n_estimators
        for i_iter in range(n_estimators):

            sampler = BalancingSampler()


            X_sampled, y_sampled = sampler.fit_resample(
                X,
                y,
                y_predict_proba=self.y_pred_proba,
                y_predict_proba_list=self.y_pred_proba_list,
                epoch=i_iter,
                all_epoch=n_estimators,
            )

            estimator = self._make_estimator(append=True)
            estimator.fit(X_sampled, y_sampled)


            if self.y_pred_proba is None:
                self.y_pred_proba = estimator.predict_proba(X)
            else:
                self.y_pred_proba = (self.y_pred_proba * i_iter + estimator.predict_proba(X)
                    ) / (i_iter + 1)

            self.y_pred_proba_list.append(estimator.predict_proba(X))
            self.estimators_features_.append(self.features_)

        return self

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from imbens.metrics import geometric_mean_score, sensitivity_score, specificity_score


    X, y = make_classification(n_classes=2, class_sep=2,
                               weights=[0.9, 0.1], n_informative=3, n_redundant=1,
                               flip_y=0, n_features=20, n_clusters_per_class=1,
                               n_samples=1000, random_state=10)

    np.random.seed(42)
    shuffle_index = np.random.permutation(len(y))
    X = X[shuffle_index]
    y = y[shuffle_index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    be = BalancedEnsembleClassifier(n_estimators=100, random_state=42)
    be.fit(X_train, y_train)

    y_pred = be.predict(X_train)

    print("Gmean:", geometric_mean_score(y_train, y_pred))
    print("sen:", sensitivity_score(y_train, y_pred))
    print("spe:", specificity_score(y_train, y_pred))

