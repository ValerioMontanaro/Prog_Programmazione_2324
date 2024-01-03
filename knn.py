import numpy as np
import pandas as pd
import splitterFactory


class KNN:

    # definizione del default constructor, k è il numero di vicini da considerare.
    def __init__(self, k: int):
        self.k = k

    def train(self, df_train: pd.DataFrame) -> (np.ndarray, np.ndarray, int):
        # prima di tutto è importante salvare gli id delle righe per risalire ai valori delle features durante i test
        # con reset_index() aggiugno una colonna alla fine con gli identificatori delle righe del dataframe
        df_train['row_id'] = df_train.reset_index().index

        # ora però l'ultima colonna è il row_id quando in realtà per coerenza dovrebbe essere la colonna target, ovvero
        # i valori y da stimare in fase di test
        # per questo motivo sposto la colonna target in ultima posizione
        cols = list(df_train.columns)
        # in questo modo sposto la colonna target in ultima posizione
        # cols[:-2] arriva a prendere tutte le colonne tranne le ultime due
        # cols[-1] prende l'ultima colonna che sarà il row_id
        # cols[-2] prende la penultima colonna che sarà la colonna target
        # riassemblo le colonne in questo modo e riscrivo il dataframe
        cols = cols[:-2] + [cols[-1]] + [cols[-2]]
        df_train = df_train[cols]

        # in quest modo memorizzo in X_train sotto forma di numpy array i valori del dataframe df_train
        X_train = df_train.iloc[:, :-1].values

        # in y_train memorizzo in un numpy array i valori della colonna target del dataframe df_train
        y_train = df_train.iloc[:, -1].values

        return X_train, y_train, self.k

    def test(self, df_test: pd.DataFrame, x_train: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
        # in questo modo memorizzo in X_test sotto forma di numpy array i valori del dataframe df_test. Di questi
        # valori si dovrà calcolare la distanza euclidea con i valori di x_train

        # per far si che il calcolo sia coerente il numero di colonne dei due array deve essere uguale
        X_test = df_test.iloc[:, :-1].values
        if x_train.shape[1] != X_test.shape[1]:
            raise ValueError("Il numero di colonne di X_train e X_test deve essere uguale")

        # calcolo ora la distanza tra ogni punto che sta in X_test e ogni punto che sta in x_train
        # in questo modo dist sarà un array di dimensione (numero di righe di X_test, numero di righe di x_train)
        dist = None  # inizializzo dist a None per evitare un warning
        for row in X_test:
            dist = np.linalg.norm(x_train - row, axis=1)  # axis=1 indica che l'operazione di norma è per ogni RIGA

        # ora dovrò considere i "k nearest neighbors" di ogni punto di X_test (ovvero considero ogni riga di dist)
        # e la sorto in modo da avere in prima posizione la distanza più piccola e in ultima la distaza più grande
        dist_sorted = sorted(dist, axis=1)  # axis=1 indica che l'operazione di sort è per ogni RIGA
        k_near_neighbors = dist_sorted[:, :k]  # prendo solo i primi k valori di ogni riga
