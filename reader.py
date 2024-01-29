from abc import ABC, abstractmethod

# classe astratta "Reader" definisce l'interfaccia che ogni classe concreta reader deve ereditare
class Reader(ABC):

    # metodo astratto "parse" definisce la firma di un metodo che essere implementato concretamente
    # il quale riceve in input un file di tipo generico
    # e restituisce un dataframe in output
    @abstractmethod
    def parse(self, filename):
        pass
