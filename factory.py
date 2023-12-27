#la classe Factory ha il compito di istanziare la giusta classe in base al tipo di file di input
from reader_csv import ReaderCSV

class Factory:
    def createReader(self, filename):
        if filename.endswith('csv'):
            return ReaderCSV()
        else:
            raise RuntimeError("Unknown log file format")