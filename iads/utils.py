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
import jax
import jax.numpy as jnp
from jax import jit
import random

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


def genere_train_test(desc_set, label_set, n_pos, n_neg):
    """permet de générer une base d'apprentissage et une base de test
    desc_set: ndarray avec des descriptions
    label_set: ndarray avec les labels correspondants
    n_pos: nombre d'exemples de label +1 à mettre dans la base d'apprentissage
    n_neg: nombre d'exemples de label -1 à mettre dans la base d'apprentissage
    Hypothèses:
       - desc_set et label_set ont le même nombre de lignes)
       - n_pos et n_neg, ainsi que leur somme, sont inférieurs à n (le nombre d'exemples dans desc_set)
    """

    # Indice du premier élément de la classe -1
    first_pos_i = label_set.tolist().index(1)
    print(first_pos_i)

    # Sélection aléatoire des indices de la base d'apprentissage
    neg_i = random.sample([i for i in range(0, first_pos_i)], n_pos)
    pos_i = random.sample([i for i in range(first_pos_i, desc_set.shape[0])], n_neg)
    appr_i = pos_i + neg_i

    # Indices de la base de test
    test_i = list(set(range(desc_set.shape[0])) - set(appr_i))

    # Création des bases d'apprentissage et de test
    train_desc = desc_set[appr_i]
    train_label = label_set[appr_i]
    test_desc = desc_set[test_i]
    test_label = label_set[test_i]

    return (train_desc, train_label), (test_desc, test_label)


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

    # on gère la seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # fréquence de chaque classe
    freq = freq_dataset(data, Y_attr)

    idx = []
    for y, f in freq.items():
        # on récupère les idx des exemples ayant la classe y
        y_idx = np.where(data["target"] == y)[0]
        rng.shuffle(y_idx)

        # on prend la même proportion que dans le dataset d'origine
        N = int(f * approx_size)
        y_idx = y_idx[:N]

        # on ajoute ces index à la liste
        idx = idx + list(y_idx)

    rng.shuffle(idx)
    return data.iloc[idx]


def df2array(df, col, index_mots) -> np.array:
    """
    Transforme un dataset de mots en un dataset de bitvec
    """

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


def dist_cosinus(u, v):
    """
    Calcule la distance cosinus entre u et v
    d=1 - <u,v>/(N(u)N(v))
    """

    N = np.linalg.norm(u) * np.linalg.norm(v)

    if N == 0:
        return 1

    return 1 - (u @ v) / N


def dist_hamming(u, v):
    """
    Calcule la distance hamming entre u et v
    Pre: u et v sont des vecteurs de bits
    """

    return np.sum(u != v)


### Utilisation de jax pour paralléliser les calculs

@jit
def calcul_eigens(data):
    """
    Calcul des valeurs propres et des vecteurs propres
    """
    data_jax = jnp.array(data)

    data_centered = data_jax - jnp.mean(data_jax, axis=0)

    cov = jnp.cov(data_centered, rowvar=False)
    cov = (cov + cov.T) / 2

    # utilisation de eigh pour exploiter la symétrie
    # cov peut devenir hermitienne à cause d'erreurs de calcul donc il faut l'indiquer à Jax
    lam, V = jnp.linalg.eigh(cov)

    return lam, V, data_centered


def pickNBestIdx(lam, N):
    """
    Choisit les N correspondant aux N valeurs propres les plus grande en valeur absolue
    """
    return jnp.flip(jnp.argsort(jnp.abs(lam)))[:N]


@jit
def projection(data_centered, V):
    """
    Projette les données sur l'espace V
    """

    # projection orthogonale sur le sous-espace
    return data_centered @ V.T


def projectionND(data_centered, lam, V, N):
    """
    Projette des données centrées sur ses N meilleurs composantes (PCA)
    """

    # on récupère les N meilleurs
    idx = pickNBestIdx(lam, N)
    best = V[idx]

    # calcul de la projection
    proj = projection(data_centered, best)

    return proj

def generate_train_test_MULTI(X, Y, train_ratio=0.8, seed=None):
    """
    Génère un ensemble d'apprentissage et un ensemble test
    """

    # on gère la seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    labels = np.unique(Y)

    # déclaration des variables qui seront initialisées dans la boucle:
    df_train = None
    df_test = None
    for l in labels:
        nb_total = X['target'].value_counts()[l]
        nb_pris = int(nb_total * train_ratio)
        print(f"Nombre d'exemples du label {l} pris pour apprendre: {nb_pris}")

        # on récupère les indices de label l
        idx = X[X['target'] == l].index.to_list()
        rng.shuffle(idx)

        idx_train = idx[:nb_pris]
        idx_test = idx[nb_pris:]

        if df_train is None:
            df_train = X.loc[idx_train]
            df_test = X.loc[idx_test]
        else:
            df_train = pd.concat([df_train, X.loc[idx_train]])
            df_test = pd.concat([df_test, X.loc[idx_test]])

    return df_train, df_test