import pandas as pd
from reader import Reader

#implementazione della classe concreta "ReaderCSV" (classe figlia della classe astratta "Reader")
class ReaderCSV(Reader):

    #implementazione concreta del metodo "parse" che riceve in input il nome del file da leggere e restituisce un dataframe
    def parse(self, filename):
        df=pd.read_csv(filename)
        return df