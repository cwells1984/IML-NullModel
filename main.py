# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd

import dataprep
import preprocessing
import processing
import nullmodelclassifier
import nullmodelregression


# Debug function gives info about data set class distribution
def debug_df_strat(df, label_column, s):
    n_entries = len(df)
    print(s)
    print(f'entries= {n_entries}')
    for label in df[label_column].unique():
        n_label = len(df.loc[df[label_column] == label])
        print(f'{label}= {n_label / n_entries}%')


# Task 7:
# Runs the experiment, performing k-fold cross validation
def perform_experiment_classifier(desc, df_original, label_column, trn_frac, k_folds):

    # First divide the data into training (trn_frac)%, and validation (remaining)% sets
    df_trn, df_val = preprocessing.df_stratified_split(df_original, label_column, trn_frac)

    # Next, divide the training set into k stratified partitions
    df_trn_partitions = preprocessing.df_stratified_partition(df_trn, label_column, k_folds)

    # Hyperparameter tuning: construct k models each using k-1 of the partitions
    best_model = processing.tune_classification_model(df_trn_partitions, df_val,
                                                      nullmodelclassifier.NullModelClassifier(), label_column)

    # Now that we have found the best model, train on k-1 and test on the remaining k-fold, getting the average score
    scores = processing.classify_cross_validation(df_trn_partitions, best_model, label_column)

    # Print the result
    print(f"{desc}: Avg classification score= {np.mean(scores)*100:2f}%, Std Dev= {np.std(scores):2f}")


# Task 7:
# Runs the experiment, performing k-fold cross validation
def perform_experiment_regression(desc, df_original, label_column, trn_frac, k_folds):

    # First divide the data into training (trn_frac)%, and validation (remaining)% sets
    df_trn, df_val = preprocessing.df_split(df_original, label_column, trn_frac)

    # Next, divide the training set into k stratified partitions
    df_trn_partitions = preprocessing.df_partition(df_trn, k_folds)

    # Hyperparameter tuning: construct k models each using k-1 of the partitions
    best_model = processing.tune_regression_model(df_trn_partitions, df_val,
                                                      nullmodelregression.NullModelRegression(), label_column)

    # Now that we have found the best model, train on k-1 and test on the last k-fold, getting the average score
    scores = processing.regression_cross_validation(df_trn_partitions, best_model, label_column)

    print(f"{desc}: Avg MSE= {np.mean(scores):2f}, Std Dev= {np.std(scores):2f}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Video Part 1
    # Demonstrate imputation of missing values
    print("IMPUTATION")
    df_breast = dataprep.prepare_breast_cancer_wisconsin('datasets/breast-cancer-wisconsin.data', display_mode=True)
    print("==============================\n")

    # Video Part 2
    # Demonstrate One-Hot Encoding
    print("ONE-HOT-ENCODING")
    df_abalone = dataprep.prepare_abalone('datasets/abalone.data', display_mode=True)
    print("==============================\n")

    # Video Part 3
    # Demonstrate Discretization by width
    print("DISCRETIZATION - WIDTH")
    df_forest = dataprep.prepare_forestfires('datasets/forestfires.data')
    print("Original DMC")
    print(df_forest['DMC'].head())
    print("Discretized by width (20 bins of equal width)")
    preprocessing.discretize_equal_width(df_forest, 'DMC', 20, range(20))
    print(df_forest['DMC'].head())
    print("==============================\n")

    # Video Part 4
    # Demonstrate Discretization by frequency
    print("DISCRETIZATION - FREQUENCY")
    df_forest = dataprep.prepare_forestfires('datasets/forestfires.data')
    print("Original DMC")
    print(df_forest['DMC'].head())
    print("Discretized by frequency (minimal frequency to most)")
    preprocessing.discretize_equal_freq(df_forest, 'DMC', 5, ['minimal', 'below avg', 'avg', 'above avg', 'most'])
    print(df_forest['DMC'].head())
    print("==============================\n")

    # Video Part 5
    # Demonstrate Standardization
    print("Z-SCORE STANDARDIZATION")
    trn_set = [1, 2, 3, 4, 5] # the mean and stdev of this set is 3 and ~1.41
    tst_set = [1, 1]
    print(trn_set)
    print(tst_set)
    z_trn_set, z_tst_set = preprocessing.standardize_z_score(trn_set, tst_set)

    # z-scored, the training set should be [-1.41, -0.71, 0, 0.71, 1.41]
    print(z_trn_set)

    # z-scored, the test set should be [-1.41, -1.41]
    print(z_tst_set)
    print("==============================\n")

    # Video Part 6
    # Read the datasets for the experiment's execution
    print("BEGIN PROCESSING")
    df_abalone = dataprep.prepare_abalone('datasets/abalone.data')
    df_breast = dataprep.prepare_breast_cancer_wisconsin('datasets/breast-cancer-wisconsin.data')
    df_car = dataprep.prepare_car('datasets/car.data')
    df_forest = dataprep.prepare_forestfires('datasets/forestfires.data')
    df_house = dataprep.prepare_house_votes_84('datasets/house-votes-84.data')
    df_machine = dataprep.prepare_machine('datasets/machine.data')

    # Perform the experiment on each dataset
    print("ABALONE")
    perform_experiment_regression("Abalone", df_abalone, 'Rings', 0.8, 5)
    print("==============================\n")

    print("BREAST CANCER")
    perform_experiment_classifier("Breast Cancer", df_breast, 'Class', 0.8, 5)
    print("==============================\n")

    print("CAR")
    perform_experiment_classifier("Car", df_car, 'CAR', 0.8, 5)
    print("==============================\n")

    print("FOREST FIRES")
    perform_experiment_regression("Forest Fires", df_forest, 'area', 0.8, 5)
    print("==============================\n")

    print("HOUSE VOTES")
    perform_experiment_classifier("House Votes", df_house, 'party', 0.8, 5)
    print("==============================\n")

    print("MACHINE")
    perform_experiment_regression("Machine", df_machine, 'PRP', 0.8, 5)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
