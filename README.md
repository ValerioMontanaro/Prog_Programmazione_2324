# AI per la Classificazione dei Tumori al Seno (Breast Cancer Classification)  

## Descrizione
Questo progetto impiega tecniche di machine learning per distinguere i tumori benigni da quelli maligni, sfruttando il dataset [Breast Cancer Wisconsin (Original)](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)). L'approccio adottato permette l'uso della Stratified Cross Validation e del metodo di Holdout per una valutazione accurata del modello. Il fulcro di questa analisi è l'impiego del classificatore k-NN (k-Nearest Neighbors) per la classificazione dei tumori.

Il programma è progettato per offrire agli utenti una varietà di opzioni di input, permettendo così una personalizzazione dettagliata dell'esecuzione e dell'analisi. Questa flessibilità si riflette nella presentazione dei risultati, che possono essere esplorati attraverso due principali modalità: la generazione di file di output dettagliati e la visualizzazione di grafici esplicativi. Questi strumenti consentono agli utenti di interpretare in maniera efficace le prestazioni del modello, fornendo così intuizioni preziose nell'ambito della diagnostica dei tumori.

## Il Dataset `breast_cancer.csv`

Il dataset `breast_cancer.csv` utilizzato in questo progetto è fondamentale per l'analisi e la classificazione dei tumori. Ecco una panoramica dettagliata del dataset:

- **Numero di Campioni**: 683 campioni.
- **Numero di Caratteristiche**: 11 caratteristiche per campione.

- **Nomi delle Caratteristiche**:
  - `Sample code number`: Identificativo unico per ogni campione.
  - `Clump Thickness`: Spessore del grumo di cellule.
  - `Uniformity of Cell Size`: Uniformità delle dimensioni cellulari.
  - `Uniformity of Cell Shape`: Uniformità delle forme cellulari.
  - `Marginal Adhesion`: Adesione marginale delle cellule.
  - `Single Epithelial Cell Size`: Dimensione della singola cellula epiteliale.
  - `Bare Nuclei`: Nuclei scoperti.
  - `Bland Chromatin`: Cromatina blanda.
  - `Normal Nucleoli`: Nucleoli normali.
  - `Mitoses`: Tasso di mitosi.
  - `Class`: Classificazione del tumore (2 per benigno, 4 per maligno).
  
- **Anteprima del Dataset**: Le prime righe del dataset sono le seguenti:

  | Sample Code Number | Clump Thickness | Uniformity of Cell Size | Uniformity of Cell Shape | Marginal Adhesion | Single Epithelial Cell Size | Bare Nuclei | Bland Chromatin | Normal Nucleoli | Mitoses | Class |
  |--------------------|-----------------|-------------------------|--------------------------|-------------------|-----------------------------|-------------|-----------------|-----------------|---------|-------|
  | 1000025            | 5               | 1                       | 1                        | 2                 | 1                           | 3           | 1               | 1               | 1       | 2     |
  | 1002945            | 5               | 4                       | 4                        | 5                 | 7                           | 10          | 3               | 2               | 1       | 2     |
  | 1015425            | 3               | 1                       | 1                        | 1                 | 2                           | 2           | 3               | 1               | 1       | 2     |
  | ...                | ...             | ...                     | ...                      | ...               | ...                         | ...         | ...             | ...             | ...     | ...   |



## Come Eseguire il Codice
Per eseguire il codice di questo progetto, seguire questi passi:

1. **Installazione delle Dipendenze**: Prima di eseguire il programma, è necessario installare le dipendenze. Questo può essere fatto eseguendo il comando `pip install -r requirements.txt` nella directory principale del progetto. Questo comando installerà tutte le librerie necessarie, come numpy, pandas, matplotlib, etc. 

2. **Caricamento del Dataset**: Il dataset `breast_cancer.csv` è già incluso nel repository e pronto per essere utilizzato. Non è necessario scaricare o preparare ulteriormente il dataset.

3. **Configurazione delle Opzioni di Input**:
    Il programma offre diverse opzioni di input per personalizzare l'esecuzione e l'analisi:
   
   - **File di input**: È possibile specificare il path di un file CSV di input contenente il dataset. Nel caso in cui non venisse specificato alcun file oppure che il path specificato non sia valido, il programma utilizzerà il file `breast_cancer.csv` incluso nel repository (ovvero il file di default).

   - **Numero di vicini (`k`) da utilizzare nel classificatore k-NN**: Questo parametro influenza il modo in cui il modello classifica i nuovi dati basandosi sui dati di addestramento. Un valore più alto di `k` può ridurre il rumore ma aumenta il rischio di perdere dettagli importanti.
  
   - **Modalità di Valutazione**: Scegliere tra le seguenti opzioni:
     - `Holdout`: Questa modalità divide il dataset randomicamente in due parti: una per l'addestramento e una per il test. È fondamentale determinare la proporzione di dati da allocare al set di test, espressa come una percentuale `X%` o `X`, dove X rappresenta un valore compreso tra 0 e 100. La scelta di X deve essere tale da garantire che nessuno dei due insiemi (di addestramento o di test) risulti vuoto. 
     - `Stratified Cross Validation`: In questa modalità, le classi benigno (2) e maligno (4) vengono suddivise ognuna in `K` parti uguali. In ciascuna classe, ogni parte viene utilizzata una volta come set di test mentre le restanti come set di addestramento. Questo metodo assicura che ogni esperimento sia eseguito su diversi sottoinsiemi di dati, fornendo una valutazione più robusta del modello. È necessario specificare il numero di parti `K` da utilizzare per la Stratified Cross Validation, che deve essere maggiore o uguale a 2 (in modo tale da avere almeno 2 esperimenti) e inferiore al numero di campioni nella classe meno rappresentata. Questo criterio assicura che ogni set di test includa almeno un esempio di ciascuna classe, permettendo così una valutazione equa e completa del modello.
   - **Metriche di Valutazione**: Queste opzioni determinano come valutare le prestazioni del modello. Sono disponibili le seguenti metriche:
     - `Accuracy Rate`: Percentuale di predizioni corrette rispetto al totale.
     - `Error Rate`: Percentuale di predizioni errate rispetto al totale.
     - `Sensitivity`: Capacità del modello di identificare correttamente i casi positivi.
     - `Specificity`: Capacità del modello di identificare correttamente i casi negativi.
     - `Geometric Mean`: Misura l'equilibrio tra Sensitivity e Specificity.
     - `All the above` : Tutte le precedenti.

4. **Esecuzione del Programma**: Eseguire il file "main.py" specificando le opzioni di input come argomenti della linea di comando quando e come richieste.

## Visualizzazione ed Interpretazione dei Risultati
I risultati possono essere visualizzati all'interno del file "risultati.xlsx" che viene generato automaticamente in due modi principali a seconda dei due diversi casi:

-**CASO 1**: In questo caso l'utente ha scelto l'holdout come modalità di test del modello. Il file "risultati.xlsx" conterrà i seguenti fogli: 
"Metrics Results" in cui è presente un tabella a due colonne che asscocia ad ogni metrica il suo valore corrispondente,
"Prevision" in cui è presente una tabella a due colonne che associa ad ogni "Sample code number" del campione di test il valore predetto dal modello.

-**CASO 2**: In questo caso l'utente ha scelto lo Stratified Cross Validation come modalità di test del modello. Il file "risultati.xlsx" conterrà i seguenti fogli: 
"Plots" in cui è presente un grafico che riporta un box plot per rappresentare la distribuzioni di valori di ogni metrica che si vuole visualizzare,
"Prevision" in cui sono presenti tante tabelle quanti sono gli esperimenti, ogni tabella associa ad ogni "Sample code number" del campione di test il valore predetto dal modello.

Per interpretare i risultati:
- Confrontare le diverse metriche di performance per comprendere l'efficacia del classificatore.
- Analizzare l'impatto delle diverse configurazioni di split del dataset e del numero di vicini `k` sulle prestazioni del modello.

## Osservazioni importanti 
I valori della colonna "Sample code number" del dataset di partenza (Breast Cancer Wisconsin (Original)) non permettono di identificare in maniera univoca ogni campione di tumore, poichè vi sono campioni diversi con "Sample code number" uguali.
Al fine di utilizzare quest'ultima colonna come indice del dataframe corrispondente al dataset, è stato necessario modificarne i valori in modo da renderli univoci,
sostituendo tutti i valori con degli indici interi progressivi che partono da 1 e arrivano al numero di campioni presenti nel dataset.


