import pandas as pd
import numpy as np

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

if __name__ == "__main__":

    # Creazione di un DataFrame di esempio con valori mancanti
    df_test = pd.DataFrame({
        'A': [1, 2, np.nan, 5, 5, 6, 2],
        'B': [np.nan, 2, 3, 4, 7, np.nan, 7],
        'C': [np.nan, 1, np.nan, 1, 2, 3, 4]
    })

    print("DataFrame originale:")
    print(df_test)

    # Istanza della classe Filler e utilizzo del metodo fill
    filler = Filler()
    df_filled = filler.fill(df_test)

    print("\nDataFrame dopo il riempimento dei valori mancanti:")
    print(df_filled)