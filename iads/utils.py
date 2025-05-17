# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""


# Fonctions utiles
# Version de départ : Février 2025

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string

# ------------------------


def create_XOR(n, var):
    """int * float -> tuple[ndarray, ndarray]
    Hyp: n et var sont positifs
    n: nombre de points voulus
    var: variance sur chaque dimension
    """
    centerA = np.array([[-1, 1], [1, -1]])  # label -1
    centerB = np.array([[-1, -1], [1, 1]])  # label 1

    cluster1 = np.random.multivariate_normal(
        centerA[0], np.array([[var, 0], [0, var]]), n
    )
    cluster2 = np.random.multivariate_normal(
        centerA[1], np.array([[var, 0], [0, var]]), n
    )
    cluster3 = np.random.multivariate_normal(
        centerB[0], np.array([[var, 0], [0, var]]), n
    )
    cluster4 = np.random.multivariate_normal(
        centerB[1], np.array([[var, 0], [0, var]]), n
    )

    data_xor = np.concatenate([cluster1, cluster2, cluster3, cluster4], axis=0)
    label_xor = []
    for i in range(2 * n):
        label_xor.append(-1)
    for i in range(2 * n):
        label_xor.append(1)

    return data_xor, np.array(label_xor)


# ------------------------ REPRENDRE ICI LES FONCTIONS SUIVANTES DU TME 2:
# genere_dataset_uniform:


def genere_dataset_uniform(d, n, binf=-1, bsup=1):
    """
    int * int * float^2 -> tuple[ndarray, ndarray]
    Hyp: n est pair
    d: nombre de dimensions de la description
    n: nombre d'exemples de chaque classe
    les valeurs générées uniformément sont dans [binf,bsup]
    """

    desc = np.random.uniform(binf, bsup, (n * d, d))

    label = np.array([-1 for i in range(0, n)] + [+1 for i in range(0, n)])

    return (desc, label)


# genere_dataset_gaussian:
def genere_dataset_gaussian(
    positive_center, positive_sigma, negative_center, negative_sigma, nb_points
):
    """les valeurs générées suivent une loi normale
    rend un tuple (data_desc, data_labels)
    """
    # Génération des exemples de la classe -1:
    data_neg = np.random.multivariate_normal(negative_center, negative_sigma, nb_points)

    # Génération des exemples de la classe +1:
    data_pos = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)

    # Construction de la description:
    data_desc = np.vstack((data_neg, data_pos))

    # Construction des labels:
    data_labels = np.array(
        [-1 for i in range(0, nb_points)] + [+1 for i in range(0, nb_points)]
    )

    return (data_desc, data_labels)


def plot2DSet(desc, labels, nom_dataset="Dataset", avec_grid=False):
    """ndarray * ndarray * str * bool-> affichage
    nom_dataset (str): nom du dataset pour la légende
    avec_grid (bool) : True si on veut afficher la grille
    la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """

    desc_negatifs = desc[labels == -1]
    desc_positifs = desc[labels == 1]
    plt.scatter(
        desc_negatifs[:, 0],
        desc_negatifs[:, 1],
        marker="o",
        color="red",
        label="classe -1",
    )
    plt.scatter(
        desc_positifs[:, 0],
        desc_positifs[:, 1],
        marker="x",
        color="blue",
        label="classe 1",
    )
    plt.title(nom_dataset)
    plt.legend()
    if avec_grid:
        plt.grid()
    plt.show()


def plot_frontiere(desc_set, label_set, classifier, step=30):
    """desc_set * label_set * Classifier * int -> NoneType
    Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
    et plus le tracé de la frontière sera précis.
    Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax = desc_set.max(0)
    mmin = desc_set.min(0)
    x1grid, x2grid = np.meshgrid(
        np.linspace(mmin[0], mmax[0], step), np.linspace(mmin[1], mmax[1], step)
    )
    grid = np.hstack((x1grid.reshape(x1grid.size, 1), x2grid.reshape(x2grid.size, 1)))

    # calcul de la prediction pour chaque point de la grille
    res = np.array([classifier.predict(grid[i, :]) for i in range(len(grid))])
    res = res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(
        x1grid, x2grid, res, colors=["darksalmon", "skyblue"], levels=[-1000, 0, 1000]
    )


def nettoyage(s: str):
    """
    Enlève les caractères polluant de s
    """
    bad_char = set(string.punctuation + string.whitespace) - {" ", "'"}

    res = ""
    for c in s.lower():
        if c in bad_char:
            res += " "
        else:
            res += c
    return res


def text2vect(s, mots_inutiles):
    """
    Nettoie la chaine et retire les mots inutiles
    """

    return [
        w
        for w in list(filter(lambda x: x not in mots_inutiles, nettoyage(s).split(" ")))
        if w
    ]


def freq_dataset(data, Y_attr):
    """
    Calcule la fréquence de chaque classe du dataset
    """

    Y, counts = np.unique(data[Y_attr], return_counts=True)

    freq = {int(y): float(f) for y, f in zip(Y, counts / len(data))}

    return freq


def sample_dataset(data, Y_attr, approx_size, seed=None):
    """
    Génère un échantillon d'une taille d'environ approx_size du dataset data en conservant la proportion des classes
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    freq = freq_dataset(data, Y_attr)

    idx = []
    for y, f in freq.items():
        y_idx = np.where(data["target"] == y)[0]
        rng.shuffle(y_idx)

        # on prend la même proportion que dans le dataset d'origine
        N = int(f * approx_size)
        y_idx = y_idx[:N]

        idx = idx + list(y_idx)

    rng.shuffle(idx)
    return data.iloc[idx]

def df2array(df, col, index_mots) -> np.array:
    res = np.zeros((len(df), len(index_mots)))
    for i in range(len(df)):
        for mot in df[col].iloc[i]:
            if mot in index_mots:
                res[i][index_mots.index(mot)] += 1

    return res

def makeWordIndex(words):
    """Make a word index from a list of words (pandas column)"""
    word_index = set()
    for i in range(len(words)):
        e = words.iloc[i]
        for w in np.unique(e):
            word_index.add(str(w))

    return list(word_index)

def word2bitmap(index, word_vec):
    """
    Transform a word vector into a bitmap
    """

    # on créer un tableau de bool indiquant la présence puis on convertit en int
    return np.isin(index, word_vec).astype(int)