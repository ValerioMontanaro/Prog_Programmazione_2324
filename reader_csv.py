#implementazione della classe concreta ReaderCSV (classe figlia della classe astratta Reader) il suo unico compito è quello di leggere file csv
#trasformarli in dataframe e restituirli pronti per il pre-processing

import pandas as pd
from reader import Reader

class ReaderCSV(Reader):

    def parse(self, filename):
        df=pd.read_csv(filename)
        return df