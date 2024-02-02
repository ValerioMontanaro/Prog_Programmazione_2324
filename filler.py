import pandas as pd

"""
la classe "Filler" ha il compito di riempire i valori mancanti di un dataframe
"""
class Filler:

    def fill(self, df):
        """
        :param df: un dataframe con potenziali valori mancanti
        :return df: un dataframe senza valori mancanti
        """

        for col in df.columns:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0] # Il metodo mode() restituisce una series, quindi selezioniamo il primo elemento, cioè il più frequente in assoluto.
                df[col].fillna(mode_value, inplace=True)

        return df