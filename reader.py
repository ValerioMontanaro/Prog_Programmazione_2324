# in questo file è definita l'interfaccia della classe reader che ha il compito di di leggere file di input nei vari formati
from abc import ABC, abstractmethod

class Reader(ABC):

    @abstractmethod
    def parse(self, filename):
        pass
