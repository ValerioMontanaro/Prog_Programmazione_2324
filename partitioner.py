import pandas as pd

"""
la classe "Partitioner" ha il compito di dividere il dataframe in due dataframe: uno contenente le features e l'altro contenente le labels
"""

class Partitioner:

    def partition(self, df):
        """
        :param df: dataframe
        :return X: dataframe delle feature con identificativo
        :return Y: dataframe della classe target con identificativo
        """

        X = df.drop(columns=['Class'])  # Dataframe delle feature con identificativo
        Y = df[['Sample code number', 'Class']] # Dataframe della classe target con identificativo

        return X, Y