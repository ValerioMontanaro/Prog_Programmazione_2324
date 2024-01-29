import numpy as np
import pandas as pd


class KNN:

    # definizione del default constructor, k è il numero di vicini da considerare.
    def __init__(self, k: int):
        self.k = k

    def train(self, df_train: pd.DataFrame) -> (np.ndarray, np.ndarray, int):
        # in quest pezzo di codice memorizzo in X_train sotto forma di numpy array i valori del dataframe df_train
        # gli tolgo l'ultima colonna perchè è la colonna target
        X_train = df_train.iloc[:, :-1].values
        # NB la prima colonna sarà il sample number che dovrà essere 'ignorato' nel momento in cui si calcola la
        # calcola distanza euclidea tra i punti di train e i punti di test
        X_train_y = df_train.values  # prendo tutto il dataframe df_train

        # in y_train memorizzo in un numpy array i valori della colonna target del dataframe df_train
        y_train = df_train.iloc[:, -1].values
        return X_train, X_train_y, y_train, self.k

    def test(self, df_test: pd.DataFrame, X_train: np.ndarray, X_train_y: np.ndarray, y_train: np.ndarray, k: int):
        # exit_df avrà come numero di colonne le colonne del dataframe di test meno la colonna target ma più una colonna
        # che conterrà l'id del punto di test che sto considerando e più un'altra che conterrà la classe predetta
        exit_df = None

        # in questo modo memorizzo in X_test sotto forma di numpy array i valori del dataframe df_test.
        # gli tolgo l'ultima colonna perchè è la colonna target

        X_test = df_test.iloc[:, :-1].values

        # per far si che il calcolo sia coerente il numero di colonne dei due numpy array deve essere uguale
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError("Il numero di colonne di X_train e X_test deve essere uguale")

        # calcolo ora la distanza tra ogni punto che sta in X_test e ogni punto che sta in x_train iterativamente per
        # ogni punto singolamrnte di X_test.
        # in questo modo dist sarà un array con tante colonne quanti punti in X_train e una riga
        for row in X_test:
            dist = np.linalg.norm(X_train[:, 1:] - row[1:], axis=1) # axis=1 dice che l'operazione di norma è per ogni RIGA
            # voglio aggingere una riga a dist che contiene i valori della colonna target corrispondente alla riga di
            # X_train che sta in dist
            dist = np.vstack((dist, y_train))
            # ora dist ha 2 righe e tante colonne quanti punti (ovvero righe) in X_train

            # voglio aggiungere un'altra riga a dist che contiene i valori dell'id corrispondente alla riga di X_train
            # che è racchiuso nella PRIMA colonna di X_train_y
            dist = np.vstack((dist, X_train_y[:, 0]))
            # ora dist ha 3 righe e tante colonne quanti punti ho in X_train

            # voglio adesso sortare la prima riga di dist portando con se la seconda riga e anche la terza
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
            row_to_be_added = np.concatenate((row, [most_frequent]))
            # NB a vstack devo passare una tupla perchè prende un positional argument
            exit_df = np.vstack((exit_df, row_to_be_added)) if exit_df is not None else np.array(row_to_be_added)

        return exit_df

