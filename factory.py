from reader_csv import ReaderCSV

#la classe "Factory" ha il compito di istanziare la giusta classe in base al tipo di file di input
class Factory:

    #Il metodo "createReader" riceve in input il nome del file da leggere e in base all'estensione del file crea l'oggetto reader corretto
    def createReader(self, filename):
        if filename.endswith('csv'):
            return ReaderCSV()
        else:
            raise RuntimeError("Unknown log file format")