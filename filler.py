import pandas as pd

#la classe "Filler" ha il compito di riempire i valori mancanti di un dataframe
class Filler:

    #il metodo "fill" riceve in input un dataframe e restituisce in output lo steso dataframe riempendo
    #eventuali valori mancanti con la moda dei valori sulla stessa colonna
    def fill(self, df):
        for col in df.columns:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0] # Il metodo mode() restituisce una series, quindi selezioniamo il primo elemento.
                df[col].fillna(mode_value, inplace=True)

        return df