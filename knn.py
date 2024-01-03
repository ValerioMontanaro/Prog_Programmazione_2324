import numpy as np
import pandas as pd
import splitterFactory


class KNN:

    # definizione del default constructor, k è il numero di vicini da considerare.
    def __init__(self, k: int):
        self.k = k

    def train(self, df_train: pd.DataFrame) -> (np.ndarray, np.ndarray, int):
        # in quest modo memorizzo in X_train sotto forma di numpy array i valori del dataframe df_train
        X_train = df_train.iloc[:, :-1].values

        # in y_train memorizzo in un numpy array i valori della colonna target del dataframe df_train
        y_train = df_train.iloc[:, -1].values

        return X_train, y_train, self.k


