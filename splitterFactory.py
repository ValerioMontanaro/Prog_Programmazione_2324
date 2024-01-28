from holdout import Holdout
from stratifiedCrossValidation import StratifiedCrossValidation

# Questa classe ha il compito di istanziare la giusta classe (estensione di validation) in base al metodo di split scelto dall'utente


class SplitterFactory:
    # Questo metodo astratto analizza il nome del metodo di split scelto dall'utente e in base a questo crea l'oggetto splitter corretto
    # input: -il nome del metodo di split
    # output: -un'istanza della corretta classe concreta che estende la classe astratta Validation
    @staticmethod
    def create_splitter(method):
        # Verifica che il metodo sia uno dei metodi supportati
        if method not in ["holdout", "stratified cross validation"]:
            raise ValueError("Metodo non supportato. Scegli tra 'holdout' e 'stratified cross validation'")

        if method == "holdout":
            return Holdout()
        elif method == "stratified cross validation":
            return StratifiedCrossValidation()
