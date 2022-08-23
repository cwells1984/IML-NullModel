import copy
import eval
import numpy as np
import pandas as pd


# Task 7:
# Train a model on k-1 partitions and test on the remaining k-fold for each k-fold, returning an array of scores
def classify_cross_validation(df_trn_partitions, model, label_column):
    scores = []
    k_folds = len(df_trn_partitions)

    for i in range(k_folds):

        # Construct a dataset for the model (k-1 partitions)
        df_trn_fold = pd.DataFrame(columns=df_trn_partitions[0].columns)
        parts_in_fold = []
        for j in range(k_folds):
            if i != j:
                df_trn_fold = pd.concat([df_trn_fold, df_trn_partitions[j]])
                parts_in_fold += [j]

        # Use the final, ith dataset for test
        df_test = df_trn_partitions[i]

        # Test the model and record its score
        y_pred = model.fit_predict(df_trn_fold, df_test, label_column)
        y_truth = df_test[label_column].values.ravel()

        # Print details about this fold
        score = eval.eval_classification_score(y_truth, y_pred)
        print(f'Fold {i+1}: Training on partitions {parts_in_fold} ({len(df_trn_fold)} entries), Testing on partition {i} ({len(df_test)} entries), Acc= {score*100:2f}%')
        scores += [score]

    return scores


# Task 7:
# Train a model on k-1 partitions and test on the remaining k-fold for each k-fold, returning an array of scores
def regression_cross_validation(df_trn_partitions, model, label_column):
    scores = []
    k_folds = len(df_trn_partitions)

    for i in range(k_folds):

        # Construct a dataset for the model (k-1 partitions)
        df_trn_fold = pd.DataFrame(columns=df_trn_partitions[0].columns)
        parts_in_fold = []
        for j in range(k_folds):
            if i != j:
                df_trn_fold = pd.concat([df_trn_fold, df_trn_partitions[j]])
                parts_in_fold += [j]

        # Use the final, ith dataset for test
        df_test = df_trn_partitions[i]

        # Test the model and record its score
        y_pred = model.fit_predict(df_trn_fold, df_test, label_column)
        y_truth = df_test[label_column].values.ravel()

        # Print details about this fold
        score = eval.eval_mse(y_truth, y_pred)
        print(f'Fold {i+1}: Training on partitions {parts_in_fold} ({len(df_trn_fold)} entries), Testing on partition {i} ({len(df_test)} entries), MSE= {score:2f}')
        scores += [score]

    return scores


# Task 7:
# Hyperparameter tuning: construct k models each using k-1 of the partitions and verify against the validation set
# Returns the model with the best parameters
def tune_classification_model(df_trn_partitions, df_val, model_to_use, label_column):
    k_folds = len(df_trn_partitions)
    best_model = None
    best_model_perf = 0
    for i in range(k_folds):

        # Construct a dataset for the model (k-1 partitions)
        df_tuning = pd.DataFrame(columns=df_trn_partitions[0].columns)
        for j in range(k_folds):
            if i != j:
                df_tuning = pd.concat([df_tuning, df_trn_partitions[j]])

        # Test the model and designate it the best if it outperforms the current best
        model = copy.deepcopy(model_to_use)
        y_pred = model.fit_predict(df_tuning, df_val, label_column)
        y_truth = df_val[label_column].values.ravel()
        score = eval.eval_classification_score(y_truth, y_pred)
        if score > best_model_perf:
            best_model = model
            best_model_perf = score

    return best_model


# Hyperparameter tuning: construct k models each using k-1 of the partitions and verify them against the validation set
# Returns the model with the best performance
def tune_regression_model(df_trn_partitions, df_val, model_to_use, label_column):
    k_folds = len(df_trn_partitions)
    best_model = None
    best_model_perf = np.inf
    for i in range(k_folds):

        # Construct a dataset for the model (k-1 partitions)
        df_tuning = pd.DataFrame(columns=df_trn_partitions[0].columns)
        for j in range(k_folds):
            if i != j:
                df_tuning = pd.concat([df_tuning, df_trn_partitions[j]])

        # Test the model and designate it the best if it outperforms the current best (has a lower MSE)
        model = copy.deepcopy(model_to_use)
        y_pred = model.fit_predict(df_tuning, df_val, label_column)
        y_truth = df_val[label_column].values.ravel()
        score = eval.eval_mse(y_truth, y_pred)
        if score < best_model_perf:
            best_model = model
            best_model_perf = score

    return best_model
