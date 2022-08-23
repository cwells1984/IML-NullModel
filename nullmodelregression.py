import numpy as np
import pandas as pd


class NullModelRegression:

    # Task 9:
    # Given a training and test dataset, calculates the mean as output
    # Returns an np array
    def fit_predict(self, df_trn, df_test, label_column):
        mean = df_trn[label_column].mean()
        return np.full((len(df_test),1), mean)
