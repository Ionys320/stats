# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import matplotlib.cm as cm



# ------------------------

def normalisation(df):
    """
    Normalise un dataframe entre 0 et 1 avec la méthode min-max
    """

    df_min = df.min()
    df_max = df.max()

    # on remplace les 0 par 1 pour pas diviser par 0
    interval = (df_max - df_min).replace(0,1)

    return (df-df_min)/interval

def dist_euclidienne(u,v):
    """
    Calcule la distance euclidienne définie par
    dist(u,v)=||u-v||_2
    """
    return np.linalg.norm(u-v)

def centroide(cluster):
    """
    Calcule le centre de gravité des exemples du dataframe, définit par
    C = (e1 + e2 + ... + eN)/N
    """

    # Cela revient à faire la moyenne sur chaque colonne
    return np.mean(cluster, axis=0)

def initialise_CHA(df):
    """
    Renvoi une partition contenant autant d'entrées que d'exemples dans df
    Où df est une base d'apprentissage
    """

    return {i: [i] for i in range(len(df)) }

def fusionne(df, p0, dist, verbose=False):
    """
    La distance utilisée permet de fusionner selon le bon linkage

    Renvoi un tuple (p1, k1, k2, d) où:
    - p1 est une partition issue de la fusion des deux clusters les plus proches
    - k1 et k2 sont les anciennes clés de ces deux clusters
    - d est la distance entre ces deux clusters
    """


    # calcul des distances de chaque couple de cluster
    distances = np.array([[k1, k2,dist(df.iloc[c1_idx], df.iloc[c2_idx])] for k1, c1_idx in p0.items() for k2, c2_idx in p0.items() if k1 < k2])

    # on récupère les clés des deux clusters les plus proche
    min_idx = np.argmin(distances[:,2])
    k1,k2,d = distances[min_idx]
    k1, k2 = int(k1), int(k2)

    # on construit p1 en filtrant k1 et k2 puis en y insérant le nouveau cluster avec la clé max(p0) + 1
    p1 = {k:cluster for k, cluster in p0.items() if k not in {k1, k2}}
    new_key = max(p0)+1
    p1[new_key] = p0[k1] + p0[k2]

    if verbose:
        print(f"fusionne: distance minimale trouvée entre [{k1}, {k2}] = {d}")
        print(f"fusionne: les 2 clusters dont les clés sont [{k1}, {k2}] sont fusionnés")
        print(f"fusionne: on cree la nouvelle clé {new_key} dans le dictionnaire")
        print(f"fusionne: les clés de [{k1}, {k2}] sont supprimées car leurs clusters ont été fusionnés.")

    return p1, k1, k2, d



def CHA_centroid(df, verbose=False, dendrogramme=False):
    """
    Renvoi une liste de liste de la forme [[k1, k2, d, S]]
    """

    def dist_centroides(cluster1, cluster2):
        """
        Calcule la distance euclidienne entre deux clusters à partir de leurs centroides
        """

        c1 = centroide(cluster1)
        c2 = centroide(cluster2)

        return dist_euclidienne(c1, c2)

    if verbose:
        print("CHA_centroid: clustering hiérarchique ascendant, version Centroid Linkage")

    res = []
    p0 = initialise_CHA(df)
    while len(p0) > 1:

        # fusion des deux clusters les plus proches
        p1, k1, k2, d =fusionne(df, p0, dist_centroides, verbose=verbose)
        k_new = max(p0)+1
        res.append([int(k1), int(k2), float(d), len(p1[k_new])])
        p0 = p1

        if verbose:
            print(f"CHA_centroid: une fusion réalisée de  {k1}  avec  {k2} de distance  {d}")
            print(f"CHA_centroid: le nouveau cluster contient {len(p1[k_new])} exemples")

    if dendrogramme:
        # Paramètre de la fenêtre d'affichage:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme (Approche Centroid Linkage', fontsize=25)
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res,
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()

    return res



def CHA_complete(df, verbose=False, dendrogramme=False):
    """
    Renvoi une liste de liste de la forme [[k1, k2, d, S]]
    """

    def distance_complete(cluster1, cluster2):
        """
        d(cluster1, cluster2) = max{ d(a,b) : a dans cluster1 et b dans cluster2 }
        """

        return np.max([dist_euclidienne(cluster1.iloc[i], cluster2.iloc[j]) for i in range(len(cluster1)) for j in range(len(cluster2)) ])


    if verbose:
        print("CHA_centroid: clustering hiérarchique ascendant, version Complete Linkage")

    res = []
    p0 = initialise_CHA(df)
    while len(p0) > 1:

        # fusion des deux clusters les plus proches
        p1, k1, k2, d =fusionne(df, p0, distance_complete, verbose)
        k_new = max(p0)+1
        res.append([int(k1), int(k2), float(d), len(p1[k_new])])
        p0 = p1

        if verbose:
            print(f"CHA_centroid: une fusion réalisée de  {k1}  avec  {k2} de distance  {d}")
            print(f"CHA_centroid: le nouveau cluster contient {len(p1[k_new])} exemples")

    if dendrogramme:
        # Paramètre de la fenêtre d'affichage:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme (Approche Complete Linkage', fontsize=25)
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res,
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()

    return res


def CHA_simple(df, verbose=False, dendrogramme=False):
    """
    Renvoi une liste de liste de la forme [[k1, k2, d, S]]
    """

    def distance_simple(cluster1, cluster2):
        """
        d(cluster1, cluster2) = min{ d(a,b) : a dans cluster1 et b dans cluster2 }
        """

        return np.min([dist_euclidienne(cluster1.iloc[i], cluster2.iloc[j]) for i in range(len(cluster1)) for j in range(len(cluster2)) ])


    if verbose:
        print("CHA_centroid: clustering hiérarchique ascendant, version Simple Linkage")

    res = []
    p0 = initialise_CHA(df)
    while len(p0) > 1:

        # fusion des deux clusters les plus proches
        p1, k1, k2, d =fusionne(df, p0, distance_simple, verbose)
        k_new = max(p0)+1
        res.append([int(k1), int(k2), float(d), len(p1[k_new])])
        p0 = p1

        if verbose:
            print(f"CHA_centroid: une fusion réalisée de  {k1}  avec  {k2} de distance  {d}")
            print(f"CHA_centroid: le nouveau cluster contient {len(p1[k_new])} exemples")

    if dendrogramme:
        # Paramètre de la fenêtre d'affichage:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme (Approche Simple Linkage', fontsize=25)
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res,
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()

    return res



def CHA_average(df, verbose=False, dendrogramme=False):
    """
    Renvoi une liste de liste de la forme [[k1, k2, d, S]]
    """

    def distance_average(cluster1, cluster2):
        """
        d(cluster1, cluster2) = max{ d(a,b) : a dans cluster1 et b dans cluster2 }
        """

        return np.mean([dist_euclidienne(cluster1.iloc[i], cluster2.iloc[j]) for i in range(len(cluster1)) for j in range(len(cluster2)) ])


    if verbose:
        print("CHA_centroid: clustering hiérarchique ascendant, version Average Linkage")

    res = []
    p0 = initialise_CHA(df)
    while len(p0) > 1:

        # fusion des deux clusters les plus proches
        p1, k1, k2, d =fusionne(df, p0, distance_average, verbose)
        k_new = max(p0)+1
        res.append([int(k1), int(k2), float(d), len(p1[k_new])])
        p0 = p1

        if verbose:
            print(f"CHA_centroid: une fusion réalisée de  {k1}  avec  {k2} de distance  {d}")
            print(f"CHA_centroid: le nouveau cluster contient {len(p1[k_new])} exemples")

    if dendrogramme:
        # Paramètre de la fenêtre d'affichage:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme (Approche Average Linkage)', fontsize=25)
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res,
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()

    return res


def CHA(DF,linkage='centroid', verbose=False,dendrogramme=False):
    """  ##### donner une documentation à cette fonction
    """
    match linkage:
        case "centroid":
            CHA_centroid(DF, verbose=verbose, dendrogramme=dendrogramme)
        case "complete":
            CHA_complete(DF, verbose=verbose, dendrogramme=dendrogramme)
        case "simple":
            CHA_simple(DF, verbose=verbose, dendrogramme=dendrogramme)
        case "average":
            CHA_average(DF, verbose=False, dendrogramme=dendrogramme)

# K-moyennes

def inertie_cluster(Ens):
    """ Array -> float
        Ens: array qui représente un cluster
        Hypothèse: len(Ens)> >= 2
        L'inertie est la somme (au carré) des distances des points au centroide.
    """

    centers = centroide(Ens)

    return np.sum([dist_euclidienne(Ens.iloc[i], centers)**2 for i in range(len(Ens))])


def init_kmeans(K,Ens):
    """ int * Array -> Array
        K : entier >1 et <=n (le nombre d'exemples de Ens)
        Ens: Array contenant n exemples
    """
    N = len(Ens)
    idx = [i for i in range(N)]
    np.random.shuffle(idx)

    idx_centroide = np.random.choice(idx,K, replace=False)

    return np.array(Ens.iloc[idx_centroide])


def plus_proche(Exe,Centres):
    """ Array * Array -> int
        Exe : Array contenant un exemple
        Centres : Array contenant les K centres
    """

    d = [dist_euclidienne(Exe, c) for c in Centres]
    return np.argmin(d)


def affecte_cluster(Base,Centres):
    """ Array * Array -> dict[int,list[int]]
        Base: Array contenant la base d'apprentissage
        Centres : Array contenant des centroides
    """

    res = {k : [] for k in range(len(Centres))}
    for i in range(len(Base)):
        k = plus_proche(Base.iloc[i],Centres)
        res[k].append(i)

    return res


def nouveaux_centroides(Base, U):
    """ Array * dict[int,list[int]] -> DataFrame
        Base : Array contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """

    res = []
    for k in U:
        c1, c2 = centroide(Base.iloc[U[k]])
        res.append([c1, c2])

    return np.array(res)


def inertie_globale(Base, U):
    """ Array * dict[int,list[int]] -> float
        Base : Array pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """

    return np.sum([inertie_cluster(Base.iloc[U[k]]) for k in U])


def kmoyennes(K, Base, epsilon, iter_max):
    """ int * Array * float * int -> tuple(Array, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : Array pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """

    # initialisation
    C = init_kmeans(K, Base)
    P0 = affecte_cluster(Base, C)
    inertie0 = inertie_globale(Base, P0)

    C = nouveaux_centroides(Base, P0)
    P1 = affecte_cluster(Base, C)
    inertie1 = inertie_globale(Base, P1)

    # distance entre les deux inerties
    d = abs(inertie1 - inertie0)

    it = 0
    while d > epsilon and it < iter_max:
        # on garde les valeurs précédentes
        P0 = P1
        inertie0 = inertie1

        # calcul d'une nouvelle partition et de son inertie
        C = nouveaux_centroides(Base, P0)
        P1 = affecte_cluster(Base, C)
        inertie1 = inertie_globale(Base, P1)

        d = abs(inertie1 - inertie0)
        it += 1

        print(f"Iteration {it} Inertie: {inertie1} Difference: {d}")

    return np.array(C), P1


def affiche_resultat(Base, Centres, Affect):
    """ DataFrame **2 * dict[int,list[int]] -> None
    """
    K = len(Centres)

    #  on transforme le colormap en couleurs utilisable par plt.scatter:
    couleurs = cm.tab20(np.linspace(0, 1, K))

    # pour chaque exemple de chaque cluster, on
    for k, c in zip(range(K), couleurs):

        for i in range(len(Affect[k])):
            idx = Affect[k][i]
            x, y = Base.iloc[idx]

            plt.scatter(x, y, color=c)

    # on met une croix sur chaque centroid
    plt.scatter(Centres[:, 0], Centres[:, 1], color='r', marker='x')
















