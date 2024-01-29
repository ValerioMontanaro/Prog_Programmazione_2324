import pandas as pd
from reader import Reader

# implementazione della classe concreta "ReaderCSV" (classe figlia della classe astratta "Reader")
class ReaderCSV(Reader):

    # implementazione concreta del metodo "parse" della classe astratta "Reader"
    # leggere il relativo commento per ulteriori informazioni sul metodo astratto "parse"
    # con la seguente implementazione concreta si definisce l'estensione "csv" del file di input
    def parse(self, filename):
        df = pd.read_csv(filename, index_col="Sample code number")
        return df