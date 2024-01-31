# Funzione: test delle classi concrete di validation
from knn import KNN
from splitterFactory import SplitterFactory
from filler import Filler
from standardizer import Standardizer
from readerFactory import ReaderFactory

factory = ReaderFactory()
reader = factory.createReader("breast_cancer.csv")
df = reader.parse("breast_cancer.csv")
filler = Filler()
df = filler.fill(df)
standardizer = Standardizer()
df_test = standardizer.standardize(df)

print(df_test)

# Presentazione all'utente delle opzioni di split e richiesta di una scelta numerica
print("Scegli il metodo di split desiderato:")
print("1: Holdout")
print("2: Stratified Cross Validation")
method_choice = input("Inserisci il numero del metodo di split (1 o 2): ").strip()

# Associa la scelta numerica all'opzione di split corrispondente
if method_choice == "1":
    method_input = "holdout"
elif method_choice == "2":
    method_input = "stratified cross validation"
else:
    raise ValueError("Scelta non valida. Inserisci '1' per Holdout o '2' per Stratified Cross Validation.")

# Utilizzo del metodo di split istanziato dal factory in base alla scelta dell'utente
splitter = SplitterFactory.create_splitter(method_input)

folds = None

if method_input == "holdout":
    # Chiedere all'utente la dimensione del test set come stringa (ad esempio, "20%")
    test_size_str = input("Inserisci la dimensione del test set (ad esempio, 20%): ")

    # Rimuovere il simbolo '%' se presente e convertire in float
    test_size_str = test_size_str.replace('%', '')

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

# a prescindere dal metodo di split scelto, folds è una LISTA che contiene TUPLE di 2 DATAFRAME (train e test)
# se il metodo è holdout avrò un solo 'folds' che conterrà una tupla di 2 dataframe (train e test)
# se il metodo è stratified cross validation avrò tanti 'folds' quanti sono i folds e ogni 'folds' conterrà una tupla di
# 2 dataframe (train e test)
for i in range(len(folds)):
    df_test_train = folds[i][0]
    df_test_test = folds[i][1]

    print(f"Dataframe di train n.{i}: \n {df_test_train}")
    print(f"Dataframe di test n.{i}: \n {df_test_test}")

    classifier = KNN(k=5)
    X_train, X_train_y, y_train = classifier.train(df_test_train)
    df_predict, df_test_adj = classifier.test(df_test_test, X_train, X_train_y, y_train)

    print(f"Dataframe predizione n.{i}: \n {df_predict}")
    print(f"Dataframe adj n.{i}: \n {df_test_adj}")

    # Estrai l'ultima colonna da entrambi i DataFrame
    last_column_df_predict = df_predict.iloc[:, -1]
    last_column_df_test_adj = df_test_adj.iloc[:, -1]

    # Confronta le colonne
    differences = last_column_df_predict != last_column_df_test_adj

    # Estrai le righe dove ci sono differenze
    different_rows = df_predict[differences]

    # Visualizza le differenze
    print("Righe con predizioni differenti:")
    print(different_rows)
