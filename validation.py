from abc import ABC, abstractmethod

import pandas as pd

"""
Questa è una classe astratta che definisce l'interfaccia sui diversi metodi di validation
che deve essere implementata dalle classi concrete che la estendono
"""


class Validation(ABC):

    @abstractmethod
    def split(self, df: pd.DataFrame) -> list:
        """
        Questo metodo astratto deve essere implementato dalle classi concrete per i diversi metodi di split
        :param df: il dataframe da dividere in set di training e testing
        :return: una lista contenente le tuple di training e testing set
        """

        pass
