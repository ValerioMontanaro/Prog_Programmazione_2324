from validation import Validation
import numpy as np
import pandas as pd

# Questa è la classe concreta che estende la classe astratta Validation
# ed implementa il metodo validate per la validazione del dataset mediante holdout

class Holdout(Validation):

    # Questo metodo implementa la validazione del dataset mediante holdout
    # riceve in ingresso:
    # df: il DataFrame che deve essere diviso in set di training e testing.
    # test_size: una proporzione (fra 0 e 1) che indica la frazione del DataFrame da utilizzare come set di test. Di default è impostata al 20% (0.2).
    # random_state: un valore opzionale per inizializzare il generatore di numeri casuali, utile per ottenere risultati riproducibili. Di default è None, il che significa che non c'è una inizializzazione deterministica.
    # e
    # restituisce: una tupla contenente il set di training e il set di test.
    def split(self, df, test_size=0.2, random_state=None):
        if random_state is not None:
            np.random.seed(random_state) # inizializzazione del generatore di numeri casuali

        shuffled_indices = np.random.permutation(len(df)) # permutazione casuale degli indici del DataFrame
        test_set_size = int(len(df) * test_size) # arrotondamento all'intero più vicino della dimensione del test set
        test_indices = shuffled_indices[:test_set_size] # selezione degli indici del test set
        train_indices = shuffled_indices[test_set_size:] # selezione degli indici del training set

        return df.iloc[train_indices], df.iloc[test_indices] # restituzione tupla dei 2 dataframe: training set e test set
