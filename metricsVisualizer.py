import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from metrics import Metrics
from openpyxl import load_workbook
from xlsxwriter.utility import xl_rowcol_to_cell

# La classe "MetricsVisualizer" ha il compito di visualizzare le metriche di valutazione della classificazione e i risultati delle previsioni, l'utente può scegliere se visualizzare una singola metrica o tutte le metriche disponibili,
# in input bisogna sempre avere la stringa "method_input" che indica il tipo di valutazione (holdout o stratified cross validation), in aggiunta ci sono dei parametri aggiuntivi che dipendono dal tipo di caso in cui ci si trova
# ci sono due casi principali:
#-CASO 1: l'utente ha scelto di effettuare la valutazione tramite holdout in questo caso bisogna avere in input due dataframe: df1 e df2
# che rappresentano rispettivamente il df delle etichette reali e il df delle previsioni.
# In questo caso i risultati delle previsioni e le metriche scelte vengono salvati in un unico file excel ("risultati.xlsx") contenente due fogli separati; "Metrics Results" che conterrà una tabella
# che associa ad ogni metrica scelta il suo valore e "Predictions" che conterrà una tabella con le previsioni e il Sample code number per record del test.
#-CASO 2: l'utente ha scelto di effettuare la valutazione tramite stratified cross validation in questo caso bisogna avere in input un array di coppie di dataframe: df1 (df2 in questo caso rimane nullo)
# in cui ogni elemento è costituito da una coppia (tupla) di dataframe dello specifico esperimento : il primo dataframe rappresenta il le etichette reali e il secondo dataframe rappresenta il le previsioni
# il numero di elementi dell'array df1 corrisponde al numero di esperimenti effettuati.
# In questo caso i risultati delle previsioni e le metriche scelte vengono salvati in un unico file excel ("risultati.xlsx")  contenente due fogli separati; "Plots" contenente un grafico con all'interno un box plot dedicato ad ogni metrica scelta
# e "Prevision" contentente tante tabelle quanti sono gli esperimenti, in ognuna di esse vengono associati il Sample code number con la relativa previsione effettuata.

class MetricsVisualizer:

       def __init__(self, method_input, df1):

           print("Selezionare una delle seguenti opzioni per la visualizzazione delle metriche:\n 1) Accuracy rate\n 2) Error rate\n 3) Sensitivity\n 4) Specificity\n 5) Geometric Mean\n 6) All the above")
           scelta = input()

           if method_input == "holdout":
               df1 = df1[0][0]
               df2 = df1[0][1]

               metrics_result = self.calculate1_metric(scelta, df1, df2)
               self.save_to_excel( metrics_result, scelta, df2)

           elif method_input == "stratified cross validation":

               results = []  # Inizializza una lista vuota per i risultati

               for true_labels_df, prediction_df  in df1:
                   metrics_result = self.calculate1_metric(scelta, true_labels_df, prediction_df)
                   results.append(metrics_result)

               self.plot_metrics(results, scelta, df1)

       # Questa funzione riceve in input la scelta dell'utente, il df delle etichette reali e il df delle previsioni e restituisce il valore della metrica scelta
       def calculate1_metric (self, scelta, df1, df2):

           metrics = Metrics()
           metrics.get_metrics(df1, df2)

           if scelta == "Accuracy rate":
               accuracy = metrics.accuracy()
               return accuracy

           if scelta == "Error rate":
               error_rate = metrics.error_rate()
               return error_rate

           if scelta == "Sensitivity":
               sensitivity = metrics.sensitivity()
               return sensitivity

           if scelta == "Specificity":
               specitivity = metrics.specificity()
               return specitivity

           if scelta == "Geometric Mean":
               geometric_mean = metrics.geometric_mean()
               return geometric_mean

           if scelta == "All the above":
               accuracy = metrics.accuracy()
               error_rate = metrics.error_rate()
               sensitivity = metrics.sensitivity()
               specificity = metrics.specificity()
               geometric_mean = metrics.geometric_mean()
               return accuracy, error_rate, sensitivity, specificity, geometric_mean

       # Questa funzione si occupa del salvataggio dei dati su excel solo nel CASO 1 (holdout)
       # Questa funzione riceve in input il nome del file excel in cui salvare i risultati, il valore della metrica scelta precedentemente calcolata, la metrica scelta dell'utente e il df delle previsioni
       # e salva i risultati nel file excel "risulati.xlsx" in due fogli separati: "Metrics Results" e "Predictions"
       def save_to_excel(self, metrics_result, scelta, df2):
           # Nomi delle metriche nell'ordine in cui appaiono in metrics_result
           metric_names = ["Accuracy", "Error rate", "Sensitivity", "Specificity", "Geometric Mean"]

           # Se i risultati sono una tupla (come per "All the above")
           if isinstance(metrics_result, tuple) :
               # Mappa i nomi delle metriche ai loro valori
               metrics_df = pd.DataFrame({
                   'Metric': metric_names,
                   'Value': list(metrics_result)
               })
           else:
              # Altrimenti, crea un DataFrame con una singola metrica e il suo valore
              metrics_df = pd.DataFrame({'Metric': [scelta], 'Value': [metrics_result]})

           # Usa pandas.ExcelWriter per scrivere in più fogli
           with pd.ExcelWriter("risultati.xlsx", mode='w') as writer:
               metrics_df.to_excel(writer, sheet_name='Metrics Results', index=False)
               df2.to_excel(writer, sheet_name='Predictions', index=True, index_label='Sample code number')

       # Questa funzione si occupa del salvataggio dei dati su excel solo nel CASO 2 (stratified cross validation)
       # Questa funzione riceve in input un array in cui ogni elemento corrisponde alle metriche calcolate per ogni esperimento, la scelta dell'utente riguardo alle metriche da voler visualizzare
       # e df1 ovvero l'array che contiene le coppie di dataframe (etichette reali e previsioni) per ogni esperimento
       # in output non restituisce nulla ma chiama la funzione "save_to_excel_with_plots" per il salvataggio e visualizazzione dei risultati
       def plot_metrics(self, results, scelta, df1):
           if scelta == "All the above":
               # 'results' è ora una lista di tuple, con ogni tupla contenente 5 metriche
               # Trasforma 'results' in un DataFrame per una manipolazione più semplice
               results_df = pd.DataFrame(results, columns=["Accuracy", "Error rate", "Sensitivity", "Specificity", "Geometric Mean"])

               # Crea 5 box plot, uno per metrica
               results_df.boxplot()
               plt.title("Box Plot per Ogni Metrica")
               plt.ylabel("Valori Metrica")
               #plt.xlabel("Metriche")
           else:
               # Gestione del caso di una singola metrica
               plt.boxplot(results)
               plt.title(f"Box Plot della Metrica {scelta}")
               plt.ylabel(scelta)
               plt.xlabel("Metriche")

           plt.savefig("box_plot.png", bbox_inches='tight')  # Salva il plot come immagine
           self.save_to_excel_with_plots(  "box_plot.png", df1)

       # Questa funzione si occupa del salvataggio dei dati su excel solo nel CASO 2 (stratified cross validation)
       # Questa funzione riceve in input il nome del file excel in cui salvare i risultati, il nome del file contenente il plot, il df delle previsioni
       # e salva i risultati nel file excel "risulati.xlsx" in due fogli separati: "Plots" e "Prevision"
       def save_to_excel_with_plots(self, plot_filename, df1):
           with pd.ExcelWriter("risultati.xlsx", engine='xlsxwriter') as writer:
               # Prima crea il foglio per il plot
               plot_sheet = writer.book.add_worksheet('Plots')

               # Nella cella "A1" inserisci l'immagine del plot ricevuta dalla funzione
               plot_sheet.insert_image('A1', plot_filename)

               # Poi crea il foglio per le previsioni
               predictions_sheet = writer.book.add_worksheet('Prevision')

               # Scrivi le tabelle delle previsioni nel foglio 'Prevision'
               row = 0
               for _, prediction_df in df1:

                   # Scrivo il DataFrame in Excel con l'indice visibile
                   prediction_df.to_excel(writer, sheet_name='Prevision', startrow=row, startcol=0, index=True, index_label='Sample code number')
                   row += len(prediction_df.index) + 3  # Aggiungo spazio tra le tabelle


if __name__ == "__main__":
   # Creare due dataframe di esempio:

 # test per la visualizzzazione dei risultati nel caso dell'holdout

   # DataFrame con etichette reali
   '''true_labels_df_test = pd.DataFrame({
       'Real Label': np.random.choice([2, 4], 10)  # Etichette reali casuali (2 per benigno, 4 per maligno)
   })

   # DataFrame con previsioni
   predictions_df_test = pd.DataFrame({
       'Predicted Label': np.random.choice([2, 4], 10)  # Previsioni casuali (2 per benigno, 4 per maligno)
   })

   metricsVisualizer = MetricsVisualizer("holdout", true_labels_df_test, predictions_df_test)'''

 # test per la visualizzzazione dei risultati nel caso della stratified cross validation

   # Numero di coppie di DataFrame da generare
   num_coppie = 5  # Puoi modificare questo valore a seconda del numero di set che vuoi generare

   df1 = []

   for _ in range(num_coppie):
       # Genera etichette reali casuali
       true_labels_df = pd.DataFrame({
           'Real Label': np.random.choice([2, 4], 10)  # Etichette reali casuali (2 o 4)
       })

       # Genera previsioni casuali
       predictions_df = pd.DataFrame({
           'Predicted Label': np.random.choice([2, 4], 10)  # Previsioni casuali (2 o 4)
       })

       # Aggiungi la coppia al tuo array df1
       df1.append(( true_labels_df, predictions_df))

   metricsVisualizer = MetricsVisualizer("stratified cross validation", df1)

   # Ora df1 contiene coppie di DataFrame di previsioni e etichette reali
