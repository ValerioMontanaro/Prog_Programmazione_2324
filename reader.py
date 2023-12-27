from abc import ABC, abstractmethod

# classe astratta "Reader" definisce l'interfaccia che ogni classe concreta reader deve ereditare
class Reader(ABC):

    #metodo astratto "parse", attraverso la sua firma si definisce un solo parametro di input cioè il nome del file da convertire
    @abstractmethod
    def parse(self, filename):
        pass
