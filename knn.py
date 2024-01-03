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
        # gli tolgo le ultime 2 colonne perchè sono il row_id e la colonna target
        X_train = df_train.iloc[:, :-2].values
        X_train_id_y = df_train.values  # prendo tutto il dataframe df_train

        # in y_train memorizzo in un numpy array i valori della colonna target del dataframe df_train
        y_train = df_train.iloc[:, -2].values
        return X_train, X_train_id_y, y_train, self.k

    def test(self, df_test: pd.DataFrame, X_train: np.ndarray, X_train_id_y: np.ndarray, y_train: np.ndarray, k: int):
        exit_df = np.array([])

        # in questo modo memorizzo in X_test sotto forma di numpy array i valori del dataframe df_test. Di questi
        # valori si dovrà calcolare la distanza euclidea con i valori di x_train
        # NB anche in questo caso il test dovrà tenere conto del row identifier
        df_test['row_id'] = df_test.reset_index().index
        cols = list(df_test.columns)
        cols = cols[:-2] + [cols[-1]] + [cols[-2]]
        df_test = df_test[cols]

        # per far si che il calcolo sia coerente il numero di colonne dei due array deve essere uguale
        X_test = df_test.iloc[:, :-2].values
        X_test_id_y = df_test.values  # prendo tutte le colonne tranne l'ultima che è il row_id
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError("Il numero di colonne di X_train e X_test deve essere uguale")

        # calcolo ora la distanza tra ogni punto che sta in X_test e ogni punto che sta in x_train iterativamente per
        # ogni punto singolamrnte di X_test.
        # in questo modo dist sarà un array con tante colonne quanti punti in X_train e una riga
        dist = None  # inizializzo dist a None per evitare un warning
        for row in X_test:
            dist = np.linalg.norm(X_train - row, axis=1)  # axis=1 dice che l'operazione di norma è per ogni RIGA
            # voglio aggingere una riga a dist che contiene i valori della colonna target corrispondente alla riga di
            # X_train che sta in dist
            dist = np.vstack((dist, y_train))
            # ora dist ha 2 righe e tante colonne quanti punti in X_train (ovvero tante colonne quanti punti in X_train)

            # voglio aggiungere un'altra riga a dist che contiene i valori dell'id corrispondente alla riga di X_train
            # che è racchiuso nella penultima colonna di X_train_id_y
            dist = np.vstack((dist, X_train_id_y[:, -2]))
            # ora dist ha 3 righe e tante colonne quanti punti ho in X_train

            # voglio adesso sortare la prima riga di dist portando con se la seconda riga
            # lo posso fare con la funzione np.argsort() che mi restituisce gli indici che ordinano una riga
            sorted_idx = np.argsort(dist[0])
            sorted_dist = dist[:, sorted_idx]

            # ora devo considerare solo le prime k colonne di sorted_dist
            k_nearest_neighbors = sorted_dist[:, :k]

            # a questo punto quante volte c'è un valore della colonna target e quante volte c'è l'altro
            unique_elements, counts_elements = np.unique(k_nearest_neighbors[1], return_counts=True)
            # unique_elements contiene i valori della colonna target
            # counts_elements contiene il numero di volte che compare quel valore della colonna target
            # np.argmax(counts_elements) restituisce l'indice del valore massimo di counts_elements
            most_frequent = unique_elements[np.argmax(counts_elements)]

            # ora devo aggiungere la riga a exit_df
            # la riga è composta row che in ogni iterazione è il punto di X_test che sto considerando e most_frequent
            # che è la classe predetta dal modello di classificazione
            exit_df = np.vstack((exit_df, np.array([row, most_frequent])))

        # finito il ciclio voglio aggiungere come colonna ad exit_df l'id del punto di X_test che sto considerando
        # questo è conservato in X_test_id_y[:, -2]
        exit_df = np.hstack((exit_df, X_test_id_y[:, -2]))

        return exit_df, self.k