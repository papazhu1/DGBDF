from DualGranularBalancedDeepForest import DualGranularBalancedDeepForest
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, average_precision_score
from imbens.metrics import geometric_mean_score, sensitivity_score, specificity_score
from evaluation import accuracy, f1_binary, f1_macro, f1_micro, gmean, sensitivity, specificity, roc_auc
import os

model_dict = {}
model_dict["rf"] = "RandomForestClassifier"
model_dict["et"] = "ExtraTreesClassifier"
model_dict["sp"] = "SelfPacedEnsembleClassifier"
model_dict["bc"] = "BalancedCascadeClassifier"
model_dict["brf"] = "BalancedRandomForestClassifier"
model_dict["ee"] = "EasyEnsembleClassifier"
model_dict["rusb"] = "RUSBoostClassifier"
model_dict["be"] = "BalancedEnsembleClassifier"


def get_config():
    config = {}
    config["enhancement_vector_method"] = "class_proba_vector"
    config["random_state"] = np.random.randint(0, 10000)
    config["max_layers"] = 5

    config["early_stop_rounds"] = 1
    config["if_stacking"] = True
    config["if_save_model"] = False
    config["train_evaluation"] = f1_macro
    config["estimator_configs"] = []

    for i in range(1):
        config["estimator_configs"].append(
            {"n_fold": 5, "type": model_dict["be"], "n_estimators": 20, "n_jobs": -1})
        config["estimator_configs"].append(
            {"n_fold": 5, "type": model_dict["be"], "n_estimators": 20, "n_jobs": -1})
        config["estimator_configs"].append(
            {"n_fold": 5, "type": model_dict["be"], "n_estimators": 20, "n_jobs": -1})
        config["estimator_configs"].append(
            {"n_fold": 5, "type": model_dict["be"], "n_estimators": 20, "n_jobs": -1})
    return config


if __name__ == "__main__":

    X = np.load("../dataset/drug_cell_feature.npy", allow_pickle=True)
    y = np.load("../dataset/drug_cell_label.npy", allow_pickle=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    DGBDF_weighted_layers_acc_list = []
    DGBDF_weighted_layers_auc_list = []
    DGBDF_weighted_layers_gmean_list = []
    DGBDF_weighted_layers_sen_list = []
    DGBDF_weighted_layers_spe_list = []
    DGBDF_weighted_layers_aupr_list = []
    DGBDF_weighted_layers_f1_macro_list = []

    per_layer_res = []
    per_layer_res_weighted_layers = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        config = get_config()

        DGBDF = DualGranularBalancedDeepForest(config)
        DGBDF.fit(X_train, y_train)
        per_layer_res.append(DGBDF.per_layer_res)
        per_layer_res_weighted_layers.append(DGBDF.per_layer_res_weighted_layers)
        DGBDF_pred_proba_weighted = DGBDF.predict_proba_weighted_layers(
            X_test)
        DGBDF_pred_weighted = DGBDF.category[
            np.argmax(DGBDF_pred_proba_weighted, axis=1)]

        print("DGBDF_weighted_layers acc: ", accuracy_score(y_test, DGBDF_pred_weighted))
        print("DGBDF_weighted_layers auc: ",
              roc_auc_score(y_test, DGBDF_pred_proba_weighted[:, 1]))
        print("DGBDF_weighted_layers gmean: ",
              geometric_mean_score(y_test, DGBDF_pred_weighted))
        print("DGBDF_weighted_layers sen: ", sensitivity_score(y_test, DGBDF_pred_weighted))
        print("DGBDF_weighted_layers spe: ", specificity_score(y_test, DGBDF_pred_weighted))
        print("DGBDF_weighted_layers aupr: ",
              average_precision_score(y_test, DGBDF_pred_proba_weighted[:, 1]))
        print("DGBDF_weighted_layers f1_macro: ",
              f1_score(y_test, DGBDF_pred_weighted, average="macro"))

        DGBDF_weighted_layers_acc_list.append(accuracy_score(y_test, DGBDF_pred_weighted))
        DGBDF_weighted_layers_auc_list.append(roc_auc_score(y_test, DGBDF_pred_proba_weighted[:, 1]))
        DGBDF_weighted_layers_gmean_list.append(geometric_mean_score(y_test, DGBDF_pred_weighted))
        DGBDF_weighted_layers_sen_list.append(sensitivity_score(y_test, DGBDF_pred_weighted))
        DGBDF_weighted_layers_spe_list.append(specificity_score(y_test, DGBDF_pred_weighted))
        DGBDF_weighted_layers_aupr_list.append(
            average_precision_score(y_test, DGBDF_pred_proba_weighted[:, 1]))
        DGBDF_weighted_layers_f1_macro_list.append(f1_score(y_test, DGBDF_pred_weighted, average="macro"))

    print("DGBDF weighted_layers acc mean: ", np.mean(DGBDF_weighted_layers_acc_list))
    print("DGBDF weighted_layers auc mean: ", np.mean(DGBDF_weighted_layers_auc_list))
    print("DGBDF weighted_layers gmean mean: ", np.mean(DGBDF_weighted_layers_gmean_list))
    print("DGBDF weighted_layers sen mean: ", np.mean(DGBDF_weighted_layers_sen_list))
    print("DGBDF weighted_layers spe mean: ", np.mean(DGBDF_weighted_layers_spe_list))
    print("DGBDF weighted_layers aupr mean: ", np.mean(DGBDF_weighted_layers_aupr_list))
    print("DGBDF weighted_layers f1_macro mean: ",
          np.mean(DGBDF_weighted_layers_f1_macro_list))
