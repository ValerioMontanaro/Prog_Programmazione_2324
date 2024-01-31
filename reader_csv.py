import pandas as pd
from reader import Reader

# implementazione della classe concreta "ReaderCSV" (classe figlia della classe astratta "Reader")
class ReaderCSV(Reader):

    # implementazione concreta del metodo "parse" della classe astratta "Reader"
    # leggere il relativo commento per ulteriori informazioni sul metodo astratto "parse"
    # con la seguente implementazione concreta si definisce l'estensione "csv" del file di input
    # il file letto in input come datframe viene prima ripulito dalle righe duplicate e
    # successivamente viene reimpostato la colonna "Sample code number" come indice del dataframe
    def parse(self, filename):
        df = pd.read_csv(filename)

        # Rimuovere le righe duplicate
        df = df.drop_duplicates()

        # impostare 'Sample code number' come indice
        df = df.set_index('Sample code number')

        return df
