import numpy as np

"""
La classe "Standardizer" ha il compito di effettuare il feature scaling di un dataframe utilizzando il metodo della standardizzazione
quest'ultima è stata preferita alla normalizzazione perché non è influenzata dalla presenza di eventuali outliers
"""


class Standardizer:

    @staticmethod
    def standardize(df):
        """
        Metodo statico che standardizza i valori di un dataframe
        :param df: dataframe generico
        :return df: stesso dataframe in input con i valori scalati secondo la standardizzazione
        """

        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number) and col != 'Class':  # la standardizzazione della series che indica l'entità maligna o benigna del tumore non ha senso e quindi non deve esssere effetuata
                mean_value = df[col].mean()
                std_value = df[col].std()
                df[col] = (df[col] - mean_value) / std_value

        return df
