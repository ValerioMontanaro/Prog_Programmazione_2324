from reader_csv import ReaderCSV

"""
La classe "Factory" ha il compito di istanziare la giusta classe in base al tipo di file di input
"""

class ReaderFactory:

    def createReader(self, filename):
        """
        Il metodo "createReader" riceve in input il nome del file da leggere e in base all'estensione del file crea l'oggetto reader corretto
        :param filename:
        :return ReaderCSV() or raise RuntimeError("Unknown log file format"):
        """

        if filename.endswith('csv'):
            return ReaderCSV()
        else:
            raise RuntimeError("Unknown log file format")