from abc import ABC, abstractmethod

# Questa è una classe astratta che definisce l'interfaccia sui diversi metodi di validation
# che deve essere implementata dalle classi concrete che la estendono


class Validation(ABC):

    # Questo metodo astratto deve essere implementato dalle classi concrete per i diversi metodi di split
    @abstractmethod
    def split(self, df, random_state=None):
        pass
