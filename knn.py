import numpy as np
import pandas as pd


class KNN:

    # definizione del default constructor, k è il numero di vicini da considerare.
    def __init__(self, k: int):
        self.k = k

    @staticmethod
    def train(df_train: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):

        """
        il metodo di train nel caso del KNN è una semplice memorizzazione dei dati che provengono dal dataframe di train
        in uscita ho diversi numpy array:
        :param df_train: è il dataframe di train
        :return X_train: che contiene i valori del dataframe di train senza la colonna target
        :return X_train_y: che contiene i valori del dataframe di train compresa la colonna target
        :return y_train: che contiene i valori della colonna target del dataframe di train
        """

        # in quest pezzo di codice memorizzo in X_train sotto forma di numpy array
        # gli tolgo l'ultima colonna perchè è la colonna target
        X_train = df_train.iloc[:, :-1].values

        # NB la prima colonna sarà il sample number che dovrà essere 'ignorato' nel momento in cui si calcola la
        # calcola distanza euclidea tra i punti di train e i punti di test
        X_train_y = df_train.values  # prendo tutto il dataframe df_train

        # in y_train memorizzo in un numpy array i valori della colonna target del dataframe df_train
        y_train = df_train.iloc[:, -1].values
        return X_train, X_train_y, y_train

    def test(self, df_test: pd.DataFrame, X_train: np.ndarray, X_train_y: np.ndarray, y_train: np.ndarray):

        """
        il metodo di test nel caso del KNN è un calcolo iterativo della distanza euclidea tra ogni punto di test e ogni
        punto di train.
        :param df_test: dataframe di test
        :param X_train: che contiene i valori del dataframe di train senza la colonna target
        :param X_train_y: che contiene i valori del dataframe di train compresa la colonna target
        :param y_train: che contiene i valori della colonna target del dataframe di train
        :return df_predict: dataframe con 2 colonne; Prima --> Sample ID Number,  Seconda --> Classe Predetta
        :return df_test_adj: dataframe con 2 colonne; Prima --> Sample ID Number,  Seconda --> Classe Reale
        """

        # exit_df avrà come numero di colonne le colonne del dataframe di test meno la colonna target ma più una colonna
        # che conterrà l'id del punto di test che sto considerando e più un'altra che conterrà la classe predetta
        df_predict = None
        df_test_adj = df_test.iloc[:, [0, -1]]

        # in questo modo memorizzo in X_test sotto forma di numpy array i valori del dataframe df_test.
        # gli tolgo l'ultima colonna perchè è la colonna target
        X_test = df_test.iloc[:, :-1].values

        # METTI COME INDICE DELLE COLONNE DEL DATAFRAME DI TEST LA COLONNA SAMPLE NUMBER

        # per far si che il calcolo sia coerente il numero di colonne dei due numpy array deve essere uguale
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError("Il numero di colonne di X_train e X_test deve essere uguale")

        # calcolo ora la distanza tra ogni punto che sta in X_test e ogni punto che sta in x_train iterativamente per
        # ogni punto singolamrnte di X_test.
        # in questo modo dist sarà un array con tante colonne quanti punti in X_train e una riga
        for row in X_test:
            dist = np.linalg.norm(X_train[:, 1:] - row[1:], axis=1)  # axis=1 dice che l'operazione di norma è per
            # ogni RIGA
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
            k_nearest_neighbors = sorted_dist[:, :self.k]

            # a questo punto quante volte c'è un valore della colonna target e quante volte c'è l'altro
            unique_elements, counts_elements = np.unique(k_nearest_neighbors[1], return_counts=True)
            # unique_elements contiene i valori della colonna target
            # counts_elements contiene il numero di volte che compare quel valore della colonna target
            # np.argmax(counts_elements) restituisce l'indice del valore massimo di counts_elements
            most_frequent = unique_elements[np.argmax(counts_elements)]

            # ora devo aggiungere la riga a exit_df
            # la riga è composta row che in ogni iterazione è il punto di X_test che sto considerando e most_frequent
            # che è la classe predetta dal modello di classificazione
            rtba = np.concatenate((row, [most_frequent]))
            # NB a vstack devo passare una tupla perchè prende un positional argument
            df_predict = np.vstack((df_predict, rtba)) if df_predict is not None else np.array(rtba)

        # trasformo l'array in un dataframe
        df_predict = pd.DataFrame(df_predict)
        df_predict = df_predict.iloc[:, [0, -1]]

        # servono 2 dataframe in uscita:
        # 1) dataframe con 2 colonne, Prima --> Sample ID Number Seconda --> Classe Reale
        # 2) dataframe con 2 colonne, Prima --> Sample ID Number Seconda --> Classe Predetta

        # NBB nello stratified cross validation avrò tanti dataframe in uscita quanti fold ho fatto (*2 perchè ho
        # sia il dataframe con le classi reali che quello con le classi predette)

        return df_predict, df_test_adj
