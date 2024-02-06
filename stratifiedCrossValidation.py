import numpy as np
import pandas as pd

from validation import Validation

"""
Questa è la classe concreta che estende la classe astratta Validation
e implementa il metodo split per la validazione del dataset mediante stratified cross validation
"""


class StratifiedCrossValidation(Validation):

    def __init__(self, n_folds):
        """
        Costruttore della classe StratifiedCrossValidation che inizializza il parametro n_folds della classe
        :param n_folds: numero di folds
        """

        self.n_folds = n_folds

    def split(self, df):
        """
        Il metodo split esegue la stratified cross validation sul dataset
        :param df: dataframe da dividere in set di training e test
        :return: folds: lista di tuple, ognuna delle quali contiene il set di training e il set di test
        """

        # Aggiunta della colonna 'fold' inizializzata a NaN
        df["fold"] = np.nan

        # Lista dei folds
        folds = []

        # Per ogni classe, assegno a ogni riga della classe un fold in modo equilibrato
        for label in df["Class"].unique():  # Per ogni classe
            class_indices = df[df["Class"] == label].index.tolist()  # Lista degli indici delle righe della classe

            # Divisione equilibrata degli indici per la classe tra i folds (se possibile)
            fold_sizes = [len(class_indices) // self.n_folds for _ in
                          range(self.n_folds)]  # Calcolo dimensione di ogni fold
            for i in range(len(class_indices) % self.n_folds):  # Per ogni elemento rimasto
                fold_sizes[i] += 1  # Aggiunta degli elementi rimasti in più ai primi fold
            print(f'Nella classe con valore ({label}) i fold hanno le seguenti dimensioni {fold_sizes}')

            # Assegnazione degli indici ai folds
            current = 0
            for i in range(self.n_folds):
                start, stop = current, current + fold_sizes[i]  # Indici di inizio e fine del fold
                df.loc[class_indices[start:stop], "fold"] = i  # Assegnazione del fold alle righe della classe
                current = stop  # Aggiornamento indice di partenza del prossimo fold

        # Creazione dei set di training e test per ciascun fold
        for i in range(self.n_folds):
            test_set = df[df["fold"] == i].iloc[:, :-1]  # Rimuove l'ultima colonna dal test set
            train_set = df[df["fold"] != i].iloc[:, :-1]  # Rimuove l'ultima colonna dal train set
            folds.append((train_set, test_set))

        return folds


if __name__ == '__main__':

    # Creazione di un DataFrame di esempio
    data = {
        'feature1': [789, 465, 543, 342, 342, 123, 456, 783, 987],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2],
        'Class': ['B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A']
    }
    df_test = pd.DataFrame(data)
    df_test.set_index("feature1", inplace=True)
    print(f"DataFrame di test:{df_test}")

    # Utilizzo del metodo di split istanziato dal factory in base alla scelta dell'utente
    splitter = StratifiedCrossValidation(n_folds=int(input("Inserisci il numero di esperimenti K (ad esempio, 5): ")))
    folds_test = splitter.split(df_test)

    # Stampa dei risultati
    for j, (train, test) in enumerate(folds_test):
        print(f"Fold {j + 1}:")
        print("Train indices:", train.index.tolist())
        print("Test indices:", test.index.tolist())

        print("Train set:", train.shape)
        print("Test set:", test.shape)
        print()
