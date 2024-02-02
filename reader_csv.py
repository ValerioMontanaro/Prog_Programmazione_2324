import pandas as pd
from reader import Reader

"""
Implementazione della classe concreta "ReaderCSV" (classe figlia della classe astratta "Reader")
"""

class ReaderCSV(Reader):



    def parse(self, filename):
        """
        implementazione concreta del metodo "parse" della classe astratta "Reader"
        leggere il relativo commento per ulteriori informazioni sul metodo astratto "parse"
        con la seguente implementazione concreta si definisce l'estensione "csv" del file di input
        il file letto in input come datframe viene prima ripulito dalle righe duplicate,
        vengono resi univoci i valori della colonna "Sample code number" e
        successivamente viene reimpostata la colonna "Sample code number" come indice del dataframe
        :param filename: file con estensione csv
        :return df: dataframe
        """

        df = pd.read_csv(filename)

        # Rimuovere le righe duplicate
        df = df.drop_duplicates()

        # Siccome i valori della series Sample code number non sono univoci, al fine di utilizzare questa series come indice
        # è necessario modificare i valori in modo che siano univoci, assegnando ad ogni record un valore progressivo ed unico
        df['Sample code number'] = range(1, len(df) + 1)

        # impostare la colonna 'Sample code number' come indice
        df = df.set_index('Sample code number')

        return df
