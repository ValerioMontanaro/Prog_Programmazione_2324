import pandas as pd

#la classe "Partitioner" ha il compito di dividere il dataframe in due dataframe: uno contenente le features e l'altro contenente le labels
class Partitioner:

    #il metodo "partition" riceve in input un dataframe e restituisce in output due dataframe: uno contenente le features e l'altro contenente le labels
    # tutti e due i dataframe in output conterranno anche la series "Sample code number" che identifica univocamentre ogni tumore
    def partition(self, df):
        X = df.drop(columns=['Class'])  # Dataframe delle feature con identificativo
        Y = df[['Sample code number', 'Class']] # Dataframe della classe target con identificativo

        return X, Y