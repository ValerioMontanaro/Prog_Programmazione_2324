import pandas as pd
import numpy as np


# La classe "Metrics" ha il compito di calcolare le metriche di valutazione della classificazione


class Metrics:

    # Il metodo "get_metrics" riceve in input due dataframe da due colonne ciascuno: "true_labels_df" e "predictions_df"
    # - Il primo df contiene gli id identificativi dei tumori (come indice) e la loro classe reale  (come series)
    # - Il secondo df contiene gli id identificativi dei tumori (come indice) e la loro classe predetta (benigno o maligno)
    # Il metodo inizializza :
    # - Un nuovo dataframe "data" (una variabile di classe) che contiene gli id identificativi dei tumori (come indice), la loro classe reale e la loro classe predetta (come series)
    # - Un nuovo dataframe "conf_matrix" (una variabile di classe) che contiene la matrice di confusione
    def get_metrics(self, true_labels_df, predictions_df):
        # Unisci i DataFrame sulla base dell'ID del tumore
        self.data = pd.merge(true_labels_df, predictions_df, left_index=True, right_index=True)
        self.true_labels = self.data.iloc[:, 0]  # Selezione della prima colonna
        self.predicted_labels = self.data.iloc[:, 1]  # Selezione della seconda colonna
        self.conf_matrix = self._confusion_matrix()

    # Il metodo restituisce la matrice di confusione
    def _confusion_matrix(self):
        expected_labels = [2, 4]     # Le etichette che ti aspetti

        conf_matrix = pd.crosstab(
            self.true_labels,
            self.predicted_labels,
            rownames=['Actual'],
            colnames=['Predicted'],
            dropna=False             # Non rimuovere le righe o le colonne con valori mancanti
        )

        # Riordina e riempi i valori mancanti con 0 per garantire coerenza
        # sostanzialmente se nell'esperimento mancava uno dei 4 casi della matrice di confusione, questo viene aggiunto e vengono aggiunti anche
        # gli indici corrispondenti. Questo permette di avere una forma standard della matrice di confusione a prescindere dal set di dati preso in considerazione.
        # Sto quindi forzando la matrice ad avere index 2 e 4 e colonne 2 e 4 e a riempire i valori mancanti con 0 al fine di poter calcolare le metriche in ogni caso
        conf_matrix = conf_matrix.reindex(index=expected_labels, columns=expected_labels, fill_value=0)
        return conf_matrix

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
        'Real Label': np.random.choice([2, 4], 10)  # Etichette reali casuali (2 per benigno, 4 per maligno)
    })

    # DataFrame con previsioni
    predictions_df_test = pd.DataFrame({
        'Predicted Label': np.random.choice([2, 4], 10)  # Previsioni casuali (2 per benigno, 4 per maligno)
    })

    metrics = Metrics()

    # Ottenere le varie metriche
    metrics.get_metrics(true_labels_df_test, predictions_df_test)
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


