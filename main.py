# Funzione: test delle classi concrete di validation
from knn import KNN
from splitterFactory import SplitterFactory
from filler import Filler
from standardizer import Standardizer
from readerFactory import ReaderFactory
from metricsVisualizer import MetricsVisualizer

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
while True:
    method_choice = input("Inserisci il numero del metodo di split (1 o 2): ").strip()

    if method_choice == "1":
        method_input = "holdout"

        while True:
            test_size_str = input("Inserisci la dimensione del test set (ad esempio, 20%): ")
            test_size_str = test_size_str.replace('%', '')
            try:
                test_size_percentage = float(test_size_str)
                if 0 <= test_size_percentage <= 100:
                    break
                else:
                    print("La dimensione del test set deve essere un numero tra 0 e 100.")
            except ValueError:
                print("Devi inserire un numero valido.")

        kwargs = {'test_size': test_size_percentage / 100, 'random_state': 42}
        break

    elif method_choice == "2":
        method_input = "stratified cross validation"

        while True:
            n_folds_str = input("Inserisci il numero di esperimenti K (ad esempio, 5): ")
            try:
                n_folds = int(n_folds_str)
                if n_folds >= 2:
                    break
                else:
                    print("Il numero di esperimenti K deve essere almeno 2.")
            except ValueError:
                print("Devi inserire un numero intero valido.")

        kwargs = {'n_folds': n_folds}
        break

    else:
        print("Scelta non valida. Inserisci '1' per Holdout o '2' per Stratified Cross Validation.")


# Utilizzo del metodo di split istanziato dal factory in base alla scelta dell'utente
splitter = SplitterFactory.create_splitter(method_input, **kwargs)

# Esegui lo split
folds = splitter.split(df_test)

for j, (train, test) in enumerate(folds):
    print(f"Fold {j + 1}:")
    print("Train indices:", train.index.tolist())
    print("Test indices:", test.index.tolist())

    print("Train set:", train.shape)
    print("Test set:", test.shape)
    print()

# A prescindere dal metodo di split scelto, folds è una LISTA che contiene TUPLE di 2 DATAFRAME (train e test)
# se il metodo è holdout avrò un solo 'folds' che conterrà una tupla di 2 dataframe (train e test)
# se il metodo è stratified cross validation avrò tanti 'folds' quanti sono i folds e ogni 'folds' conterrà una tupla di
# 2 dataframe (train e test)

risultati = []  # Creare una lista vuota per conservare le tuple di dataframe (df_predict, df_test_adj) di ogni fold
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

    # Salvare i dataframe df_predict e df_test_adj come una tupla nella lista risultati
    risultati.append((df_test_adj, df_predict))

    print("le dimensioni di risultati sono: ", len(risultati))

print("Selezionare una delle seguenti opzioni per la visualizzazione delle metriche:\n 1: Accuracy rate\n 2: Error rate\n 3: Sensitivity\n 4: Specificity\n 5: Geometric Mean\n 6: All the above")
while True:
    try:
        scelta = input("Inserisci il numero corrispondente alla metrica desiderata (un numero intero compreso tra 1 e 6): ").strip()
        scelta_int = int(scelta)  # Converti la scelta in un intero
        if 1 <= scelta_int <= 6:
            metrics_visualizer = MetricsVisualizer(method_input, risultati, scelta_int)
            break  # Esce dal ciclo se l'input è valido
        else:
            raise ValueError
    except ValueError:
        print("Input non valido")
