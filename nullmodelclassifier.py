import numpy as np
import pandas as pd


class NullModelClassifier:

    # Task 9:
    # Given a training and test dataset, calculates the majority class as output
    # Returns an np array
    def fit_predict(self, df_trn, df_test, label_column):
        mode = df_trn[label_column].mode()
        return np.full((len(df_test),1), mode)
