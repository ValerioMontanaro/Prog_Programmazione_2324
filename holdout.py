from validation import Validation
import numpy as np


# Questa è la classe concreta che estende la classe astratta Validation
# e implementa il metodo split per la validazione del dataset mediante holdout


class Holdout(Validation):
    # Questo metodo implementa la validazione del dataset mediante holdout
    # input:
    # -df: il DataFrame che deve essere diviso in set di training e testing.
    # -random_state: un valore opzionale per inizializzare il generatore di numeri casuali, utile per ottenere risultati riproducibili. Di default è None, il che significa che non c'è una inizializzazione deterministica.
    # -test_size: la dimensione del test set come frazione del dataset originale. Di default è 0.2, che significa che il test set sarà il 20% del dataset originale.
    # output:
    # -folds: una lista contenente una tupla, che contiene il set di training e il set di test.
    def split(self, df, random_state=None, test_size=0.2):
        folds = []

        if random_state is not None:
            np.random.seed(random_state)  # inizializzazione del generatore di numeri casuali

        shuffled_indices = np.random.permutation(len(df))  # permutazione casuale degli indici del DataFrame
        test_set_size = int(
            len(df) * test_size)  # arrotondamento all'intero più vicino della dimensione del test set
        test_indices = shuffled_indices[:test_set_size]  # selezione degli indici del test set
        train_indices = shuffled_indices[test_set_size:]  # selezione degli indici del training set

        folds.append((df.iloc[train_indices], df.iloc[test_indices]))
        return folds
