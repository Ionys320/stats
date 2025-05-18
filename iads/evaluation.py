# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd

# ------------------------
# TODO: à compléter  plus tard
# ------------------------


def analyse_perfs(L):
    """L : liste de nombres réels non vide
    rend le tuple (moyenne, écart-type)
    """
    moy = np.mean(L)
    ecart = np.std(L, ddof=1)

    return moy, ecart

#
def crossval(X, Y, n_iterations, iteration):
    # Mélange des index
    index = np.random.permutation(len(X))
    Xm = X[index]
    Ym = Y[index]

    # Extraction des indexs pour le dataset de test
    test_it_idx = np.arange(
        iteration * len(X) // n_iterations, (iteration + 1) * len(X) // n_iterations - 1
    )

    # Extraction des indexs restants pour le dataset d'apprentissage
    app_it_idx = np.setdiff1d(np.arange(len(X)), test_it_idx)

    Xapp = Xm[app_it_idx]
    Yapp = Ym[app_it_idx]
    Xtest = Xm[test_it_idx]
    Ytest = Ym[test_it_idx]

    return Xapp, Yapp, Xtest, Ytest





def crossval_strat(X, Y, n_iterations, iteration):
    # Déterminer la distribution des classes
    classes = np.unique(Y)

    # Mélange des index
    index = np.random.permutation(len(X))
    Xm = X[index]
    Ym = Y[index]

    test_indices = []

    # Pour chaque classe, sélectionner les indices appropriés
    for c in classes:
        # Trouver tous les indices de cette classe
        class_indices = np.where(Ym == c)[0]
        n_samples = len(class_indices)

        # Calculer les indices de début et de fin pour cette itération
        start_idx = (iteration * n_samples) // n_iterations
        end_idx = ((iteration + 1) * n_samples) // n_iterations

        # S'assurer que les indices sont valides
        if start_idx < end_idx and start_idx < n_samples:
            # Ajouter les indices au test set
            test_indices.extend(class_indices[start_idx:end_idx])

    # Convertir en array NumPy
    test_it_idx = np.array(test_indices, dtype=int)

    # Extraction des indexs restants pour le dataset d'apprentissage
    app_it_idx = np.setdiff1d(np.arange(len(X)), test_it_idx)

    # Créer les datasets d'apprentissage et de test
    Xapp = Xm[app_it_idx]
    Yapp = Ym[app_it_idx]
    Xtest = Xm[test_it_idx]
    Ytest = Ym[test_it_idx]

    return Xapp, Yapp, Xtest, Ytest


import copy


def validation_croisee(C, DS, nb_iter):
    """Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]"""

    X, y = DS

    perf = []

    for i in range(nb_iter):

        Xapp, Yapp, Xtest, Ytest = crossval_strat(X, y, nb_iter, i)
        classifier = copy.deepcopy(C)
        classifier.train(Xapp, Yapp)
        taux = classifier.accuracy(Xtest, Ytest)
        perf.append(taux)

        print(
            f"Itération {i}: taille base app.= {len(Yapp)}	taille base test= {len(Ytest)}	Taux de bonne classif: {taux}"
        )

    taux_moyen, taux_ecart = analyse_perfs(perf)
    return perf, taux_moyen, taux_ecart

#
def crossval(X, Y, n_iterations, iteration):
    indices_app = []
    indices_test = []

    start = iteration * (len(X) // n_iterations)
    end = (iteration + 1) * (len(X) // n_iterations) - 1

    for i in range(start, end + 1):
        indices_test.append(i)

    for i in range(len(X)):
        if i not in indices_test:
            indices_app.append(i)

    Xapp = np.array(indices_app)
    Yapp = Xapp

    Xtest = np.array(indices_test)
    Ytest = Xtest

    return X[indices_app], Y[indices_app], X[indices_test], Y[indices_test]
#
#
# def analyse_perfs(L):
#     """ L : liste de nombres réels non vide
#         rend le tuple (moyenne, écart-type)
#     """
#     return (np.mean(L), np.std(L))
#

def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """

    perf = []

    for i in range(nb_iter):
        cl = copy.deepcopy(C)
        Xapp, Yapp, Xtest, Ytest = crossval(DS[0], DS[1], nb_iter, i)
        cl.train(Xapp, Yapp)
        perf.append(cl.accuracy(Xtest, Ytest))
        print(
            f"Itération {i} : taille base app = {len(Xapp)} taille base test = {len(Xtest)} Taux de bonne classif: {cl.accuracy(Xtest, Ytest)} ")

    moyenne, ecart_type = analyse_perfs(perf)
    return perf, moyenne, ecart_type



