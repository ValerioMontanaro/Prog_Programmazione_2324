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
factory = SplitterFactory()
splitter = factory.create_splitter()
folds = splitter.split(df_test, random_state=42)

print(type(folds))
print(folds)

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

classifier = KNN(k=1)
X_train, X_train_id_y, y_train, k = classifier.train(df_test_train)
exit_df = classifier.test(df_test_test, X_train, X_train_id_y, y_train, k)

print(exit_df)



