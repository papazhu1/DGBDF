import numpy as np
from imbens.sampler import RandomUnderSampler
from imbens.sampler._under_sampling.base import BaseUnderSampler
from imbens.sampler._over_sampling.base import BaseOverSampler
from funcs import *


class BalancingSampler(BaseUnderSampler):
    def __init__(
            self,
            *,
            soft_resample=True,
            sampling_strategy="auto",

            random_state=None,
            num_bins=5,
    ):
        super().__init__()

        self.sampling_strategy = sampling_strategy
        self.soft_resample = soft_resample
        self.random_state = random_state
        self.num_bins = num_bins
        self.use_uncertainty = True
        self.use_hardness = True

    def fit_resample(self, X, y, **kwargs):
        return super().fit_resample(X, y, **kwargs)

    def _fit_resample(
            self,
            X,
            y,
            *,
            epoch,
            all_epoch,
            y_predict_proba,
            y_predict_proba_list,
            **kwargs):
        n_samples, n_features = X.shape

        if epoch == 0:
            return RandomUnderSampler().fit_resample(X, y)

        capacities = calculate_bin_capacities(
            num_bins=self.num_bins,
            t=epoch / all_epoch * 5,
            alpha=0.05)

        bins = [[] for _ in range(self.num_bins)]

        x_pos_samples = []
        x_neg_samples = []
        y_pos_samples = []
        y_neg_samples = []

        for i in range(n_samples):
            if y[i] == 1:
                x_pos_samples.append(X[i])
                y_pos_samples.append(y[i])
            else:
                x_neg_samples.append(X[i])
                y_neg_samples.append(y[i])

        index_0 = np.where(y == 0)[0]
        index_1 = np.where(y == 1)[0]

        error_0 = y_predict_proba[index_0, 0] * (-1) + 1
        error_1 = y_predict_proba[index_1, 1] * (-1) + 1

        hardness_0 = error_to_hardness(error_0)
        hardness_1 = error_to_hardness(error_1)

        index_0_sorted = np.argsort(hardness_0)
        index_1_sorted = np.argsort(hardness_1)

        x_pos_samples_sorted = np.array(x_pos_samples)[index_1_sorted]
        y_pos_samples_sorted = np.array(y_pos_samples)[index_1_sorted]
        x_neg_samples_sorted = np.array(x_neg_samples)[index_0_sorted]
        y_neg_samples_sorted = np.array(y_neg_samples)[index_0_sorted]
        index_0 = index_0[index_0_sorted]
        index_1 = index_1[index_1_sorted]

        begins = [0] + list(np.cumsum(capacities)[:-1])
        begins = [int(x * len(index_0)) for x in begins]
        ends = list(np.cumsum(capacities))
        ends = [int(x * len(index_0)) for x in ends]

        for i in range(self.num_bins):
            bins[i] = index_0[begins[i]:ends[i]]

        bin_pos = index_1
        per_bin_uncertainties = []
        bin_pos_uncertainties = []

        for i in range(self.num_bins):
            cur_bin_uncertainties = []
            for j in range(len(bins[i])):
                index = bins[i][j]
                pred_list = []
                for k in range(len(y_predict_proba_list)):
                    pred_list.append(y_predict_proba_list[k][index][0])
                uncertainty = np.var(pred_list)

                cur_bin_uncertainties.append(uncertainty)
            per_bin_uncertainties.append(cur_bin_uncertainties)

        for j in range(len(bin_pos)):
            index = bin_pos[j]
            pred_list = []
            for k in range(len(y_predict_proba_list)):
                pred_list.append(y_predict_proba_list[k][index][1])
            uncertainty = np.var(pred_list)
            bin_pos_uncertainties.append(uncertainty)

        per_bin_probabilities = []
        bin_pos_probabilities = []

        choosed_index = []

        for i in range(len(per_bin_uncertainties)):
            probabilities = uncertainty_to_probability_by_sum(per_bin_uncertainties[i])
            if self.use_uncertainty == False:
                probabilities = np.array([1 / len(bins[i]) for _ in range(len(bins[i]))])
            per_bin_probabilities.append(probabilities)

            non_zero_num = np.count_nonzero(probabilities)
            if non_zero_num < np.min([len(bins[i]), len(bin_pos) // len(bins)]):
                choosed_index.extend(np.random.choice(bins[i], non_zero_num, replace=False, p=probabilities))
                bins[i] = np.setdiff1d(bins[i], choosed_index)
                choosed_index.extend(
                    np.random.choice(bins[i], min(len(bins[i]), len(bin_pos) // len(bins)), replace=False))
            else:
                choosed_index.extend(
                    np.random.choice(bins[i], min(len(bins[i]), len(bin_pos) // len(bins)), replace=False,
                                     p=probabilities))

        bin_pos_probabilities = uncertainty_to_probability_by_sum(bin_pos_uncertainties)

        for i in range(len(choosed_index)):
            if choosed_index[i] not in index_0:
                raise ValueError("The index is not in index_0")
        choosed_index = np.concatenate([choosed_index, bin_pos], axis=0)
        np.random.shuffle(choosed_index)
        return X[choosed_index], y[choosed_index]


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = make_classification(n_classes=2, class_sep=2,
                               weights=[0.9, 0.1], n_informative=3, n_redundant=1,
                               flip_y=0, n_features=20, n_clusters_per_class=1,
                               n_samples=1000, random_state=10)

    np.random.seed(42)
    shuffle_index = np.random.permutation(len(y))
    X = X[shuffle_index]
    y = y[shuffle_index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    etc1 = ExtraTreesClassifier(n_estimators=100, random_state=42)
    etc2 = ExtraTreesClassifier(n_estimators=100, random_state=42)

    RUS = RandomUnderSampler()
    X_train_rus, y_train_rus = RUS.fit_resample(X_train, y_train)
    etc1.fit(X_train_rus, y_train_rus)
    etc2.fit(X_train_rus, y_train_rus)

    y_pred_proba_list = []
    y_pred_proba_list.append(etc1.predict_proba(X_train))
    y_pred_proba_list.append(etc2.predict_proba(X_train))
    y_pred_proba = np.mean(y_pred_proba_list, axis=0)

    sampler = BalancingSampler()
    new_sample = sampler.fit_resample(X_train,
                                      y_train,
                                      epoch=1,
                                      all_epoch=20,
                                      y_predict_proba=y_pred_proba,
                                      y_predict_proba_list=y_pred_proba_list)
