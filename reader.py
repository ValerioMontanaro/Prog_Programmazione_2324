from abc import ABC, abstractmethod

import pandas as pd

"""
classe astratta "Reader" definisce l'interfaccia che ogni classe concreta reader deve ereditare
"""


class Reader(ABC):

    @abstractmethod
    def parse(self, filename) -> pd.DataFrame:
        """
        metodo astratto "parse" definisce la firma di un metodo che essere implementato concretamente
        :param filename: input un file di tipo generico
        :return: dataframe in output
        """

        pass
