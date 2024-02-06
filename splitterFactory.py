from holdout import Holdout
from stratifiedCrossValidation import StratifiedCrossValidation

"""
Questa classe ha il compito di istanziare la giusta classe (estensione di validation) in base al metodo di split scelto dall'utente.
"""


class SplitterFactory:
    @staticmethod
    def create_splitter(method: str, **kwargs):
        """
        Questo metodo statico analizza il nome del metodo di split scelto dall'utente e in base a questo crea l'oggetto splitter corretto
        :param method: il nome del metodo di split
        :param kwargs: i parametri del metodo di split (es. test_size e random_state per holdout o n_folds per stratified cross validation)
        :return: un'istanza della corretta classe concreta che estende la classe astratta Validation
        """

        # Verifica che il metodo sia uno dei metodi supportati
        if method not in ["holdout", "stratified cross validation"]:
            raise ValueError("Metodo non supportato. Scegli tra 'holdout' e 'stratified cross validation'")

        if method == "holdout":
            return Holdout(**kwargs)
        elif method == "stratified cross validation":
            return StratifiedCrossValidation(**kwargs)
