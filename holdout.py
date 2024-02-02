from validation import Validation
import numpy as np

"""
Questa è la classe concreta che estende la classe astratta Validation
e implementa il metodo split per la validazione del dataset mediante holdout
"""


class Holdout(Validation):

    def __init__(self, test_size, random_state=None):
        """
        Costruttore della classe Holdout che inizializza i parametri test_size e random_state della classe
        :param test_size: dimensione del test set come frazione del dataset originale
        :param random_state: valore opzionale per inizializzare il generatore di numeri casuali, utile per ottenere risultati riproducibili
        """

        self.test_size = test_size
        self.random_state = random_state

    def split(self, df):
        """
        Il metodo split esegue la holdout sul dataset
        :param df: dataframe da dividere in set di training e test
        :return: folds: lista di tuple, ognuna delle quali contiene il set di training e il set di test
        """

        # Lista dei folds
        folds = []

        if self.random_state is not None:
            np.random.seed(self.random_state)  # inizializzazione del generatore di numeri casuali

        shuffled_indices = np.random.permutation(len(df))  # permutazione casuale degli indici del DataFrame
        test_set_size = int(
            len(df) * self.test_size)  # arrotondamento all'intero più vicino della dimensione del test set
        test_indices = shuffled_indices[:test_set_size]  # selezione degli indici del test set
        train_indices = shuffled_indices[test_set_size:]  # selezione degli indici del training set

        folds.append((df.iloc[train_indices], df.iloc[test_indices]))  # accesso alle righe tramite gli indici di posizione (iloc) selezionati e aggiunta delle tuple alla lista dei folds
        return folds
