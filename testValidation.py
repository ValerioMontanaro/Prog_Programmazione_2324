# Funzione: test delle classi concrete di validation
import numpy as np
import pandas as pd
from knn import KNN
from splitterFactory import SplitterFactory

# Creazione di un DataFrame di esempio
# la colonna sample number è l'identificatore della riga e non è una feature sarebbe il sample number del tumore, questa
# colonna è per scopi di test
data = {
    'sample number': [123, 321, 121, 131, 313, 333, 111, 133, 312, 122],
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'class': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
}
df_test = pd.DataFrame(data)

print(df_test)

# Utilizzo del metodo di split istanziato dal factory in base alla scelta dell'utente
method_input = input(
    "Inserisci il metodo di split desiderato.\nScegli tra 'holdout' e 'stratified cross validation': ").strip().lower()
splitter = SplitterFactory.create_splitter(method_input)

folds = None

if method_input == "holdout":
    # Chiedere all'utente la dimensione del test set come stringa (ad esempio, "20%")
    test_size_str = input("Inserisci la dimensione del test set (ad esempio, 20%): ")

    # Rimuovere il simbolo '%' e convertire in float
    if test_size_str.endswith('%'):
        test_size_str = test_size_str[:-1]
    else:
        raise ValueError("La dimensione del test set deve essere seguita dal simbolo '%'")

    try:
        test_size_percentage = float(test_size_str)
    except ValueError:
        raise ValueError("Devi inserire un numero valido")

    # Verifica che la percentuale sia tra 0 e 100
    if not 0 <= test_size_percentage <= 100:
        raise ValueError("La dimensione del test set deve essere un numero tra 0 e 100")

    # Convertire la percentuale in una frazione
    test_size_fraction = test_size_percentage / 100

    folds = splitter.split(df_test, random_state=42, test_size=test_size_fraction)


elif method_input == "stratified cross validation":
    # Chiedere all'utente il numero di folds come stringa
    n_folds_str = input("Inserisci il numero di esperimenti K (ad esempio, 5): ")

    try:
        n_folds = int(n_folds_str)
    except ValueError:
        raise ValueError("Devi inserire un numero intero valido")

    # Verifica che il numero di folds sia almeno 2
    if n_folds < 2:
        raise ValueError("Il numero di esperimenti K deve essere almeno 2")

    folds = splitter.split(df_test, n_folds=n_folds)

print(folds)
print(len(folds))

# a prescindere dal metodo di split scelto, folds è una LISTA che contiene TUPLE di 2 DATAFRAME (train e test)
# se il metodo è holdout avrò un solo 'folds' che conterrà una tupla di 2 dataframe (train e test)
# se il metodo è stratified cross validation avrò tanti 'folds' quanti sono i folds e ogni 'folds' conterrà una tupla di
# 2 dataframe (train e test)
for i in range(len(folds)):
    df_test_train = folds[i][0]
    df_test_test = folds[i][1]

    print(df_test_train)
    print(df_test_test)

    classifier = KNN(k=5)
    X_train, X_train_id_y, y_train, k = classifier.train(df_test_train)
    exit_df = classifier.test(df_test_test, X_train, X_train_id_y, y_train, k)

    print(exit_df)
