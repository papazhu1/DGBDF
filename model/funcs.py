import numpy as np
import csv


def error_to_hardness(x):
    return np.tanh(6 * (x - 0.5)) / 2 + 1


def append_results_to_csv(results,
                          csv_file="gcForest_sampled_res.csv"):
    """
    Appends a dictionary of results to a CSV file, ensuring required fields are present.

    Parameters:
    results (dict): The results dictionary to append to the CSV file.
    csv_file (str): The path to the CSV file.
    """
    required_keys = [
        "layer", "accuracy", "f1_score", "auc", "gmean", "sensitivity", "specificity",
        "aupr", "generalized_performance", "true_neg", "false_pos", "false_neg", "true_pos", "recall_0",
        "recall_1", "precision_0", "precision_1", "imbalanced_rate", "neg_num", "pos_num"
    ]


    for key in required_keys:
        if key not in results:
            raise ValueError(f"Missing required key in results dictionary: {key}")


    file_exists = False
    try:
        with open(csv_file, 'r') as file:
            file_exists = True
    except FileNotFoundError:
        file_exists = False


    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=required_keys)
        if (not file_exists) or (file.tell() == 0):
            writer.writeheader()
        writer.writerow(results)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def uncertainty_to_probability_by_power(uncertainties, power=5):

    scores = np.power(uncertainties, power)
    scores = scores / np.max(scores)
    probabilities = softmax(scores)
    return probabilities


def uncertainty_to_probability_by_sum(uncertainties):
    uncertainties = np.array(uncertainties)

    sum_uncertainty = np.sum(uncertainties)
    if sum_uncertainty == 0:
        probabilities = np.array([1 / len(uncertainties) for _ in uncertainties])
    else:
        probabilities = uncertainties / sum_uncertainty
    return probabilities


def calculate_bin_capacities(num_bins, t, alpha=0.1):
    indices = np.arange(num_bins)
    raw_capacities = np.exp(-alpha * indices * t)


    normalized_capacities = raw_capacities / raw_capacities.sum()
    return normalized_capacities



def train_K_fold_paralleling(args):
    use_uncertainty = True
    use_bucket = True
    use_resample = True

    train_index, val_index, X, y, est, buckets, bucket_variances, index_1_sorted, uncertainty_1_sorted = args
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    sampled_1_index = None
    if index_1_sorted is None or uncertainty_1_sorted[0] == np.nan:
        pass
    else:
        in_train_index_1 = np.in1d(index_1_sorted, train_index)
        if isinstance(in_train_index_1, np.ndarray) and in_train_index_1.dtype == bool:
            in_train_index_1 = np.where(in_train_index_1)[0]

        index_1_sorted = np.array(index_1_sorted)
        uncertainty_1_sorted = np.array(uncertainty_1_sorted)
        index_1_sorted = index_1_sorted[in_train_index_1]
        uncertainty_1_sorted = uncertainty_1_sorted[in_train_index_1]

        probability = uncertainty_to_probability_by_sum(uncertainty_1_sorted)
        if np.isnan(probability).any():
            print("probability[0] == np.NAN")
        else:
            if use_uncertainty is True:
                sampled_1_index = np.random.choice(index_1_sorted, int(len(index_1_sorted)), replace=True,
                                                   p=probability)

            else:
                sampled_1_index = np.random.choice(index_1_sorted, int(len(index_1_sorted)), replace=True)
    if buckets is not None:
        index_0 = np.where(y_train == 0)[0]
        index_1 = np.where(y_train == 1)[0]


        for i in range(len(buckets)):


            in_train_index = np.in1d(buckets[i], train_index)
            if isinstance(in_train_index, np.ndarray) and in_train_index.dtype == bool:
                in_train_index = np.where(in_train_index)[0]

            buckets[i] = buckets[i][in_train_index]
            if bucket_variances is not None:
                bucket_variances[i] = bucket_variances[i][in_train_index]


        index_0_selected = []
        num_empty_buckets = 0

        for i in range(len(buckets)):
            if len(buckets[i]) == 0:
                num_empty_buckets += 1
        for i in range(len(buckets)):

            if len(buckets[i]) == 0:
                continue

            if bucket_variances is not None:

                probabilities = uncertainty_to_probability_by_sum(bucket_variances[i])

                assert len(probabilities) == len(buckets[i])

                if use_uncertainty is True:
                    sampled_cur_bucket = np.random.choice(buckets[i],
                                                          int(len(index_1) // (len(buckets) - num_empty_buckets)),
                                                          replace=False,
                                                          p=probabilities)
                else:
                    sampled_cur_bucket = np.random.choice(buckets[i],
                                                          int(len(index_1) // (len(buckets) - num_empty_buckets)),
                                                          replace=False)
                index_0_selected.extend(sampled_cur_bucket)
            else:
                sampled_cur_bucket = np.random.choice(buckets[i],
                                                      int(len(index_1) // (len(buckets) - num_empty_buckets)),
                                                      replace=False)
                index_0_selected.extend(sampled_cur_bucket)

        if use_bucket is False:

            all_majority_samples = []
            for i in range(len(buckets)):
                all_majority_samples.extend(buckets[i])

            index_0_selected = np.random.choice(all_majority_samples, int(len(index_1)), replace=False)

        index_0_selected = np.array(index_0_selected)

        if sampled_1_index is not None:
            new_train_index = np.concatenate([index_0_selected, index_1, sampled_1_index])
        else:
            new_train_index = np.concatenate([index_0_selected, index_1])

        X_train, y_train = X[new_train_index], y[new_train_index]

    if use_resample is False:
        X_train, y_train = X[train_index], y[train_index]

    est.fit(X_train, y_train)
    y_proba = est.predict_proba(X_val)
    y_pred = est.predict(X_val)

    return [est, y_proba, y_pred, val_index]


def predict_proba_parallel(ars):
    estimator, x_test = ars
    return estimator.predict_proba(x_test)
