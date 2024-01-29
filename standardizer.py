import pandas as pd
import numpy as np

#La classe "Standardizer" ha il compito di effettuare il feature scaling di un dataframe utilizzando il metodo della standardizzazione
#quest'ultima è stata preferita alla normalizzazazione perche non è influenzata dalla presenza di eventuali outliers
class Standardizer:

    #il metodo "standardize" riceve in input un dataframe e restituisce in output lo stesso dataframe con i valori scalati secondo la standardizzazione
    #ogni valore viene prima sottratto della media e poi diviso per la deviazione standard
    def standardize(self, df):
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number)  and col != 'Class': #la standardizzazione dei codici identificativi del tumore non ha senso e quindi non deve esssere effetuata
                mean_value = df[col].mean()
                std_value = df[col].std()
                df[col] = (df[col] - mean_value) / std_value

        return df

