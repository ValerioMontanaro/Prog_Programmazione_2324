from validation import Validation
import numpy as np

# Questa è la classe concreta che estende la classe astratta Validation
# e implementa il metodo validate per la validazione del dataset mediante holdout


class Holdout(Validation):

    # Il costruttore implementa l'input della percentuale del test_size da parte dell'utente come float tra 0 e 1
    def __init__(self):
        # Chiedere all'utente la percentuale di test_size come stringa (ad esempio, "20%")
        test_size_str = input("Inserisci la percentuale del set di test (ad esempio, 20%): ")

        # Rimuovere il simbolo '%' e convertire in float
        if test_size_str.endswith('%'):
            test_size_str = test_size_str[:-1]
        else:
            raise ValueError("La percentuale deve essere seguita dal simbolo '%'")

        try:
            test_size_percentage = float(test_size_str)
        except ValueError:
            raise ValueError("Devi inserire un numero valido")

        # Verifica che la percentuale sia tra 0 e 100
        if not 0 <= test_size_percentage <= 100:
            raise ValueError("La percentuale del test_size deve essere un numero tra 0 e 100")

        # Convertire la percentuale in una frazione
        self.test_size = test_size_percentage / 100

    # Questo metodo implementa la validazione del dataset mediante holdout
    # input:
    # -df: il DataFrame che deve essere diviso in set di training e testing.
    # -random_state: un valore opzionale per inizializzare il generatore di numeri casuali, utile per ottenere risultati riproducibili. Di default è None, il che significa che non c'è una inizializzazione deterministica.
    # output:
    # -una tupla contenente il set di training e il set di test.
    def split(self, df, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)  # inizializzazione del generatore di numeri casuali

        shuffled_indices = np.random.permutation(len(df))  # permutazione casuale degli indici del DataFrame
        test_set_size = int(len(df) * self.test_size)  # arrotondamento all'intero più vicino della dimensione del test set
        test_indices = shuffled_indices[:test_set_size]  # selezione degli indici del test set
        train_indices = shuffled_indices[test_set_size:]  # selezione degli indici del training set

        return df.iloc[train_indices], df.iloc[test_indices]  # restituzione tupla dei 2 dataframe: training set e test set
