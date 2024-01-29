import pandas as pd
import numpy as np


# La classe "Metrics" ha il compito di calcolare le metriche di valutazione della classificazione


class Metrics:

    # Il costruttore riceve in input due dataframe da due colonne ciascuno: "true_labels_df" e "predictions_df"
    # - Il primo df contiene gli id identificativi dei tumori e la loro classe reale (benigno o maligno)
    # - Il secondo df contiene gli id identificativi dei tumori e la loro classe predetta (benigno o maligno)
    # Il metodo inizializza :
    # - Un nuovo dataframe "data" (una variabile di classe) che contiene gli id identificativi dei tumori, la loro classe reale e la loro classe predetta
    # - Un nuovo dataframe "conf_matrix" (una variabile di classe) che contiene la matrice di confusione
    def __init__(self, true_labels_df, predictions_df):
        # Unisci i DataFrame sulla base dell'ID del tumore
        self.data = pd.merge(true_labels_df, predictions_df, on='Sample code number')
        self.true_labels = self.data.iloc[:, 1]
        self.predicted_labels = self.data.iloc[:, 2]
        self.conf_matrix = self._confusion_matrix()

    # Il metodo restituisce la matrice di confusione
    def _confusion_matrix(self):
        return pd.crosstab(self.true_labels, self.predicted_labels, rownames=['Actual'], colnames=['Predicted'],
                           dropna=True)  # il nome delle righe è "Actual", il nome delle colonne è "Predicted"

    # Il metodo restituisce l'accuracy rate
    def accuracy(self):
        return np.diag(self.conf_matrix).sum() / self.conf_matrix.values.sum()

    # Il metodo restituisce l'error rate
    def error_rate(self):
        return 1 - self.accuracy()

    # Il metodo restituisce la sensitivity (True Positive Rate)
    def sensitivity(self):
        TP = self.conf_matrix[4][
            4]  # numero di casi in cui è stato predetto un tumore maligno quando il tumore è maligno
        FN = self.conf_matrix[2][
            4]  # numero di casi in cui è stato predetto un tumore benigno quando il tumore è maligno
        return TP / (TP + FN)

    # Il metodo restituisce la specificity (True Negative Rate)
    def specificity(self):
        TN = self.conf_matrix[2][
            2]  # numero di casi in cui è stato predetto un tumore benigno quando il tumore è benigno
        FP = self.conf_matrix[4][
            2]  # numero di casi in cui è stato predetto un tumore maligno quando il tumore è benigno
        return TN / (TN + FP)

    # Il metodo restituisce la geometric mean (radice quadrata del prodotto di sensitivity e specificity)
    def geometric_mean(self):
        return np.sqrt(self.sensitivity() * self.specificity())


if __name__ == "__main__":
    # Creare due dataframe di esempio:

    # DataFrame con etichette reali
    true_labels_df_test = pd.DataFrame({
        'Sample code number': np.arange(1, 11),  # ID identificativi da 1 a 10
        'Real Label': np.random.choice([2, 4], 10)  # Etichette reali casuali (2 per benigno, 4 per maligno)
    })

    # DataFrame con previsioni
    predictions_df_test = pd.DataFrame({
        'Sample code number': np.arange(1, 11),  # ID identificativi da 1 a 10
        'Predicted Label': np.random.choice([2, 4], 10)  # Previsioni casuali (2 per benigno, 4 per maligno)
    })

    metrics = Metrics(true_labels_df_test, predictions_df_test)

    # Ottenere le varie metriche
    accuracy = metrics.accuracy()
    error_rate = metrics.error_rate()
    sensitivity = metrics.sensitivity()
    specificity = metrics.specificity()
    geometric_mean = metrics.geometric_mean()

    # Stampare i dataframe
    print(true_labels_df_test)
    print(predictions_df_test)

    # Stampare i risultati
    print(f"Confusion Matrix: {metrics.conf_matrix}")

    print(f"Accuracy: {accuracy}")
    print(f"Error Rate: {error_rate}")

    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")

    print(f"Geometric Mean: {geometric_mean}")
