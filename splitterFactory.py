from holdout import Holdout
from stratifiedCrossValidation import StratifiedCrossValidation

# Questa classe ha il compito di istanziare la giusta classe in base al metodo di split scelto dall'utente


class SplitterFactory:
    def __init__(self):
        # Chiedere all'utente il metodo di split
        method_input = input(
            "Inserisci il metodo di split desiderato.\nScegli tra 'holdout' e 'stratified cross validation': ").strip().lower()

        # Verifica che il metodo sia uno dei metodi supportati
        if method_input not in ["holdout", "stratified cross validation"]:
            raise ValueError("Metodo non supportato. Scegli tra 'holdout' e 'stratified cross validation'")

        self.method = method_input

    # Questo metodo analizza il nome del metodo di split scelto dall'utente e in base a questo crea l'oggetto splitter corretto
    # output: -un'istanza della classe astratta Validation
    def create_splitter(self):
        if self.method == "holdout":
            return Holdout()
        elif self.method == "stratified cross validation":
            return StratifiedCrossValidation()
