# Funzione: test delle classi concrete di validation
import pandas as pd
from knn import KNN
from splitterFactory import SplitterFactory

# Creazione di un DataFrame di esempio
data = {
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

# Stampa dei risultati

print(type(folds))
print(folds)

train_indices = None
test_indices = None

# Stampa dei risultati
for i, (train, test) in enumerate(folds):
    print(f"Fold {i+1}:")
    train_indices = train.index.tolist()
    test_indices = test.index.tolist()
    print("Train indices:", train.index.tolist())
    print("Test indices:", test.index.tolist())

    print("Train set:", train.shape)
    print("Test set:", test.shape)

df_test_train = df_test.iloc[train_indices]
df_test_test = df_test.iloc[test_indices]

classifier = KNN(k=5)
X_train, X_train_id_y, y_train, k = classifier.train(df_test_train)
exit_df = classifier.test(df_test_test, X_train, X_train_id_y, y_train, k)

print(exit_df)
