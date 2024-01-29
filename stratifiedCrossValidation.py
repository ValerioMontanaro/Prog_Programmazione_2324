import numpy as np

from validation import Validation


# Questa è la classe concreta che estende la classe astratta Validation
# e implementa il metodo split per la validazione del dataset mediante stratified cross validation


class StratifiedCrossValidation(Validation):
    # Questo metodo implementa la validazione del dataset mediante stratified cross validation
    # input:
    # -df: il DataFrame che deve essere diviso in set di training e testing.
    # -n_folds: il numero di folds. Di default è 5.
    # output:
    # -folds: una lista contenente le n_folds tuple, ognuna delle quali contiene il set di training e il set di test.
    def split(self, df, n_folds=5):
        # Aggiunta della colonna 'fold' inizializzata a NaN
        df["fold"] = np.nan

        # Lista dei folds
        folds = []

        # Per ogni classe, assegno a ogni riga della classe un fold in modo equilibrato
        for label in df["Class"].unique():  # Per ogni classe
            class_indices = df[df["Class"] == label].index.tolist()  # Lista degli indici delle righe della classe

            # Divisione equilibrata degli indici per la classe tra i folds (se possibile)
            fold_sizes = [len(class_indices) // n_folds for _ in
                          range(n_folds)]  # Calcolo dimensione di ogni fold
            for i in range(len(class_indices) % n_folds):  # Per ogni elemento rimasto
                fold_sizes[i] += 1  # Aggiunta degli elementi rimasti in più ai primi fold

            # Assegnazione degli indici ai folds
            current = 0
            for i in range(n_folds):
                start, stop = current, current + fold_sizes[i]  # Indici di inizio e fine del fold
                df.loc[class_indices[start:stop], "fold"] = i  # Assegnazione del fold alle righe della classe
                current = stop  # Aggiornamento indice di partenza del prossimo fold

        # Creazione dei set di training e test per ciascun fold
        for i in range(n_folds):
            test_set = df[df["fold"] == i].iloc[:, :-1]  # Rimuove l'ultima colonna dal test set
            train_set = df[df["fold"] != i].iloc[:, :-1]  # Rimuove l'ultima colonna dal train set
            folds.append((train_set, test_set))

        return folds
