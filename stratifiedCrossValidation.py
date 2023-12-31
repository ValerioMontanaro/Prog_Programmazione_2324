import numpy as np

from validation import Validation


# Questa classe implementa la validazione del dataset mediante stratified cross validation


class StratifiedCrossValidation(Validation):

    # Il costruttore implementa l'input del numero di folds da parte dell'utente come intero
    def __init__(self):
        # Chiedere all'utente il numero di folds come stringa
        n_folds_str = input("Inserisci il numero di esperimenti K (ad esempio, 5): ")

        try:
            n_folds = int(n_folds_str)
        except ValueError:
            raise ValueError("Devi inserire un numero intero valido")

        # Verifica che il numero di folds sia almeno 2
        if n_folds < 2:
            raise ValueError("Il numero di esperimenti K deve essere almeno 2")

        self.n_folds = n_folds

    # Questo metodo implementa la validazione del dataset mediante stratified cross validation
    # input:
    # -df: il DataFrame che deve essere diviso in set di training e testing.
    # -random_state: un valore opzionale per inizializzare il generatore di numeri casuali, utile per ottenere risultati riproducibili. Di default è None, il che significa che non c'è una inizializzazione deterministica.
    # output:
    # -folds: una lista contenente le n_folds tuple, ognuna delle quali contiene il set di training e il set di test.
    def split(self, df, random_state=None):
        # Aggiunta della colonna 'fold' inizializzata a NaN
        df["fold"] = np.nan

        # Lista dei folds
        folds = []

        # Per ogni classe, assegno a ogni riga della classe un fold in modo equilibrato
        for label in df["class"].unique():  # Per ogni classe
            class_indices = df[df["class"] == label].index.tolist()  # Lista degli indici delle righe della classe

            # Divisione equilibrata degli indici per la classe tra i folds
            fold_sizes = [len(class_indices) // self.n_folds for _ in
                          range(self.n_folds)]  # Calcolo dimensione di ogni fold
            for i in range(len(class_indices) % self.n_folds):
                fold_sizes[i] += 1  # Aggiunta degli elementi rimasti in più ai primi fold

            # Assegnazione degli indici ai folds
            current = 0
            for i in range(self.n_folds):
                start, stop = current, current + fold_sizes[i]  # Indici di inizio e fine del fold
                df.loc[class_indices[start:stop], "fold"] = i  # Assegnazione del fold alle righe della classe
                current = stop  # Aggiornamento indice di partenza del prossimo fold

        # Creazione dei set di training e test per ciascun fold
        for i in range(self.n_folds):
            test_set = df[df["fold"] == i]
            train_set = df[df["fold"] != i]
            folds.append((train_set, test_set))

        return folds
