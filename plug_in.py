import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

def create_scorer():
    '''
        créé un scorer permettant de calculer le pourcentage
        d'erreur lié à une prédiction

        renvoie
        -------
        scorer : sklearn.metrics.scorer
                 objet callable retournant le pourcentage
                 d'erreur de prédiction pour une sortie
                 d'un estimator
    '''
    def custom_score(y_true, y_pred, **kwargs):
        y_pred = np.sign(y_pred)
        return np.count_nonzero(y_pred != y_true) / len(y_true)
    scorer = make_scorer(custom_score)
    return scorer

class PlugInClassifier(object):
    '''
        Classe générique pour la classification plug-in
    '''
    def __init__(self):
        '''
            stocke
            ------
            self.model : sklearn.linear_model.*
                         modèle linéaire de sklearn
            self.fitted : boolean
                          True si le modèle a été entraîné, False sinon
            self.scorer : sklearn.metrics.scorer
                          objet callable retournant le pourcentage
                          d'erreur de prédiction pour une sortie
                          d'un estimator
        '''
        self.model = None
        self.fitted = False
        self.scorer = create_scorer()

    def fit(self, X, Y):
        '''
            permet d'apprendre le modèle sur les données X,Y

            paramètres
            ----------
            X : np.array, shape (n_echantillons, n_dimensions)
                données d'apprentissage
            Y : np.array, shape (n_echantillons, 1)
                labels des données d'apprentissage
        '''
        self.model.fit(X,Y)
        self.fitted = True

    def score(self, X, Y):
        '''
            calcule le pourcentage d'erreur de prédiction sur X

            paramètres
            ----------
            X : np.array, shape (n_echantillons, n_dimensions)
                données d'apprentissage
            Y : np.array, shape (n_echantillons, 1)
                labels des données d'apprentissage
            renvoie
            -------
            np.count_nonzero(predictions != Y) / len(Y) : float
                taux d'erreur de prédiction (entre 0 et 1)
        '''
        predictions = np.sign(self.model.predict(X))
        return np.count_nonzero(predictions != Y) / len(Y)

    def get_weights(self):
        '''
            récupère le vecteur de poids du modèle

            renvoie
            -------
            self.model.coef_ : np.array, shape (n_features)
                               vecteur de paramètres
        '''
        assert(self.fitted == True)
        return self.model.coef_

class LinearRegressionClassifier(PlugInClassifier):
    """
        Régression linéaire adaptée à la classification binaire
        basée sur sklearn.linear_model.LinearRegression
    """
    def __init__(self):
        '''
            stocke
            ------
            self.model : sklearn.linear_model.LinearRegression
                         modèle de régression linéaire de sklearn
        '''
        super().__init__()
        self.model = LinearRegression()

class RidgeClassifier(PlugInClassifier):
    """
        Régression ridge adaptée à la classification binaire
        basée sur sklearn.linear_model.Ridge
    """
    def __init__(self, alpha=1.0, max_iter=None):
        '''
            paramètres
            ----------
            alpha : float (1.0 par défaut)
                    coefficient de régularisation
            max_iter : int (None par défaut)
                       nombre maximum d'itérations du solveur
                       du gradient conjugué
            stocke
            ------
            self.model : sklearn.linear_model.Ridge
                         modèle de régression Ridge de sklearn
                         paramétré par les valeurs ci-dessus
        '''
        super().__init__()
        self.model = Ridge(alpha=alpha, max_iter=max_iter)

class LassoClassifier(PlugInClassifier):
    """
        Algorithme du LASSO adapté à la classficiation binaire
        basée sur sklearn.linear_model.Lasso
    """
    def __init__(self, alpha=1.0, max_iter=1000):
        '''
            paramètres
            ----------
            alpha : float (1.0 par défaut)
                    coefficient de régularisation
            max_iter : int (None par défaut)
                       nombre maximum d'itérations du solveur
                       du gradient conjugué
            stocke
            ------
            self.model : sklearn.linear_model.Lasso
                         modèle de régression linéaire avec régularisation
                         Lasso de sklearn
                         paramétré par les valeurs ci-dessus
        '''
        super().__init__()
        self.model = Lasso(alpha=alpha, max_iter=max_iter)
