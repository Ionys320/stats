# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2025

# Import de packages externes
import numpy as np
import pandas as pd
import copy

import graphviz as gv

import random

from iads.Clustering import dist_euclidienne


# ---------------------------


class Classifier:
    """Classe (abstraite) pour représenter un classifieur
    Attention: cette classe est ne doit pas être instanciée.
    """

    def __init__(self, input_dimension):
        """Constructeur de Classifier
        Argument:
            - intput_dimension (int) : dimension de la description des exemples
        Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension

    def train(self, desc_set, label_set):
        """Permet d'entrainer le modele sur l'ensemble donné
        desc_set: ndarray avec des descriptions
        label_set: ndarray avec les labels correspondants
        Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        raise NotImplementedError("Please Implement this method")

    def score(self, x):
        """rend le score de prédiction sur x (valeur réelle)
        x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def predict(self, x):
        """rend la prediction sur x (soit -1 ou soit +1)
        x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """Permet de calculer la qualité du système sur un dataset donné
        desc_set: ndarray avec des descriptions
        label_set: ndarray avec les labels correspondants
        Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # Calcul du nombre d'exemples:
        nb_exemples = len(desc_set)

        # Calcul du nombre d'erreurs:
        nb_erreurs = 0

        # Parcours des exemples:
        for i in range(nb_exemples):
            # Calcul de la prédiction:
            prediction = self.predict(desc_set[i])
            # En cas d'incohérence, on incrémente le nombre d'erreurs:
            if prediction != label_set[i]:
                nb_erreurs += 1

        # Calcul de la qualité du classifieur:
        accuracy = 1 - nb_erreurs / nb_exemples

        return accuracy


class ClassifierKNN(Classifier):
    """Classe pour représenter un classifieur par K plus proches voisins.
    Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension, k, dist=dist_euclidienne):
        """Constructeur de Classifier
        Argument:
            - intput_dimension (int) : dimension d'entrée des exemples
            - k (int) : nombre de voisins à considérer
        Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self, input_dimension)
        self.k = k
        self.dist = dist

    def score(self, x):
        """rend la proportion de chaque classe parmis les k ppv de x (valeur réelle)
        x: une description : un ndarray
        """
        # Calcul des distances entre x et les exemples du dataset:
        distances = [self.dist(x, ex) for ex in self.desc_set]

        # Tri des distances
        indices = np.argsort(distances)

        # Sélection des k plus proches voisins
        k_nearest_labels = self.label_set[indices[: self.k]]

        # Calcul de la classe majoritaire parmis les k plus proche voisin
        l, counts = np.unique(k_nearest_labels, return_counts=True)

        return dict(zip(l, counts))

    def predict(self, x):
        """rend la prediction sur x (-1 ou +1)
        x: une description : un ndarray
        """
        # Calcul du score
        score = self.score(x)

        # On renvoi la classe majoritaire parmi les voisins
        return max(score, key=score.get)

    def train(self, desc_set, label_set):
        """Permet d'entrainer le modele sur l'ensemble donné
        desc_set: ndarray avec des descriptions
        label_set: ndarray avec les labels correspondants
        Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set


class ClassifierLineaireRandom(Classifier):
    """Classe pour représenter un classifieur linéaire aléatoire
    Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension):
        """Constructeur de Classifier
        Argument:
            - intput_dimension (int) : dimension de la description des exemples
        Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self, input_dimension)

        v = np.random.uniform(-1, 1, (1, input_dimension))
        self.w = v / np.linalg.norm(v)

    def train(self, desc_set, label_set):
        """Permet d'entrainer le modele sur l'ensemble donné
        desc_set: ndarray avec des descriptions
        label_set: ndarray avec les labels correspondants
        Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        print("Pas d'apprentissage pour ce classifieur")

    def score(self, x):
        """rend le score de prédiction sur x (valeur réelle)
        x: une description
        """
        return self.w @ x

    def predict(self, x):
        """rend la prediction sur x (soit -1 ou soit +1)
        x: une description
        """
        s = self.score(x)
        if s >= 0:
            return 1
        else:
            return -1


class ClassifierPerceptron(Classifier):
    """Perceptron de Rosenblatt"""

    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """Constructeur de Classifier
        Argument:
            - input_dimension (int) : dimension de la description des exemples (>0)
            - learning_rate (par défaut 0.01): epsilon
            - init est le mode d'initialisation de w:
                - si True (par défaut): initialisation à 0 de w,
                - si False : initialisation par tirage aléatoire de valeurs petites
        """
        Classifier.__init__(self, input_dimension)
        self.learning_rate = learning_rate
        self.input_dimension = input_dimension

        if init:
            self.w = np.zeros(input_dimension)
        else:
            self.w = []
            # Génération de très petits nombres (entre 0.001 et 0.001)
            for v in range(input_dimension):
                v = np.random.uniform(0.0, 1.0)
                v = 2 * v - 1  # Interval [-1, 1]
                v *= 0.001  # Interval [-0.001, 0.001]
                self.w.append(v)

        self.allw = [self.w.copy()]

    def train_step(self, desc_set, label_set):
        """Réalise une unique itération sur tous les exemples du dataset
        donné en prenant les exemples aléatoirement.
        Arguments:
            - desc_set: ndarray avec des descriptions
            - label_set: ndarray avec les labels correspondants
        """

        # Liste d'indices aléatoires avec shuffle
        indices = [i for i in range(0, desc_set.shape[0])]
        np.random.shuffle(indices)

        # Évaluation de l'erreur de prédiction pour chaque x_i
        for i in indices:
            # Mise à jour de w
            if self.predict(desc_set[i]) != label_set[i]:
                print(label_set[i], desc_set[i])
                self.w += self.learning_rate * label_set[i] * np.array(desc_set[i])

            self.allw.append(self.w.copy())

    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """Apprentissage itératif du perceptron sur le dataset donné.
        Arguments:
            - desc_set: ndarray avec des descriptions
            - label_set: ndarray avec les labels correspondants
            - nb_max (par défaut: 100) : nombre d'itérations maximale
            - seuil (par défaut: 0.001) : seuil de convergence
        Retour: la fonction rend une liste
            - liste des valeurs de norme de différences
        """
        diffs = []
        for i in range(nb_max):
            # print(i)
            w_old = self.w.copy()
            self.train_step(desc_set, label_set)

            diff = self.w - w_old
            diffs.append(diff)
            norme = np.linalg.norm(diff)

            if norme < seuil:
                break

        return diffs

    def score(self, x):
        """rend le score de prédiction sur x (valeur réelle)
        x: une description
        """
        return self.w @ x

    def predict(self, x):
        """rend la prediction sur x (soit -1 ou soit +1)
        x: une description
        """
        score = self.score(x)

        return 1 if score > 0 else -1

    def get_allw(self):
        return self.allw.copy()


class ClassifierPerceptronBiais(ClassifierPerceptron):
    """Perceptron de Rosenblatt avec biais
    Variante du perceptron de base
    """

    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """Constructeur de Classifier
        Argument:
            - input_dimension (int) : dimension de la description des exemples (>0)
            - learning_rate (par défaut 0.01): epsilon
            - init est le mode d'initialisation de w:
                - si True (par défaut): initialisation à 0 de w,
                - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)
        # Affichage pour information (décommentez pour la mise au point)
        # print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        self.input_dimension = input_dimension

    def train_step(self, desc_set, label_set):
        """Réalise une unique itération sur tous les exemples du dataset
        donné en prenant les exemples aléatoirement.
        Arguments:
            - desc_set: ndarray avec des descriptions
            - label_set: ndarray avec les labels correspondants
        """

        # Liste d'indices aléatoires avec shuffle
        indices = [i for i in range(0, desc_set.shape[0])]
        np.random.shuffle(indices)

        # Évaluation de l'erreur de prédiction pour chaque x_i
        for i in indices:
            # Mise à jour de w
            if self.score(desc_set[i]) * label_set[i] < 1:
                self.w += self.learning_rate * label_set[i] * desc_set[i]

            self.allw.append(self.w.copy())


class ClassifierMultiOAA(Classifier):
    """Classifieur multi-classes One-Against-All"""

    def __init__(self, cl_bin):
        """Constructeur de Classifier
        Argument:
            - cl_bin: classifieur binaire positif/négatif
        """
        super().__init__(cl_bin.input_dimension)
        self.cl_bin_template = cl_bin
        self.cl = []
        self.labels = []

    def train(self, desc_set, label_set):
        """Entraîne le modèle sur les données fournies"""

        self.labels = np.unique(label_set)
        self.cl = []

        for label in self.labels:
            Ytmp = np.where(label_set == label, 1, -1)
            clf = copy.deepcopy(self.cl_bin_template)
            clf.train(desc_set, Ytmp)
            self.cl.append(clf)

    def score(self, x):
        """Retourne le score de prédiction maximal pour x"""

        scores = [clf.score(x) for clf in self.cl]
        return scores

    def predict(self, x):
        """Prédit la classe de x (celle avec le score maximal)"""

        idx = np.argmax(self.score(x))
        return self.labels[idx]

    def accuracy(self, desc_set, label_set):
        yhat = np.array([self.predict(x) for x in desc_set])
        return np.where(label_set == yhat, 1.0, 0.0).mean()


class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self,input_dimension)  # Appel du constructeur de la classe mère
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD(desc_set, label_set, self.epsilon, self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        
        return self.racine.classifie(x)

    def number_leaves(self):
        """ rend le nombre de feuilles de l'arbre
        """
        return self.racine.compte_feuilles()
    
    def draw(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)


class ClassifierArbreNumerique(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision numérique
    """

    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None

    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision [' + str(self.dimension) + '] eps=' + str(self.epsilon)

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.racine = construit_AD_num(desc_set, label_set, self.epsilon, self.LNoms)

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass

    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok = 0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i, :]) == label_set[i]:
                nb_ok = nb_ok + 1
        acc = nb_ok / (desc_set.shape[0] * 1.0)
        return acc

    def number_leaves(self):
        """ rend le nombre de feuilles de l'arbre
        """
        return self.racine.compte_feuilles()

    def affiche(self, GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)


# ---------------------------


# -------

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    values, counts = np.unique(Y, return_counts=True)

    idx = np.argmax(counts)

    return values[idx]

def shannon(P):
    """ list[Number] -> float
        Hypothèse: P est une distribution de probabilités
        - P: distribution de probabilités
        rend la valeur de l'entropie de Shannon correspondante
    """
    k = len(P)
    if k < 2:
        return 0.0
    
    P = np.array(P)
    log_base_k = np.log(k)
    entropy = 0.0

    for p in P:
        if p > 0:
            entropy -= p * np.log(p) / log_base_k
    
    return entropy
    

def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    _, nb_fois = np.unique(Y,return_counts=True)
    nb_fois = nb_fois / len(Y)
    return shannon(nb_fois)

def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    

    entropie_ens = entropie(Y)
    if (entropie_ens <= epsilon):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui minimise l'entropie
        # min_entropie : la valeur de l'entropie minimale
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur de l'entropie de la classe pour chaque attribut.

        for i in range(len(LNoms)):
            valeurs = np.unique(X[:, i])
            entropie_attribut = 0
            for v in valeurs:
                # on récupère les observations avec v comme valeur de l'attribut X_i
                idx = np.where(X[:, i] == v)

                # p(v)
                v_freq = len(idx[0])/len(Y)

                # Y | v
                Y_v = Y[idx]

                entropie_attribut += v_freq * entropie(Y_v)
            
            # on fait une recherche de min pour pas s'embêter avec le - 
            # sinon ça serait une recherche de maximum
            if entropie_attribut < min_entropie:
                min_entropie = entropie_attribut
                i_best = i
                Xbest_valeurs = valeurs
         
       #############################################
        
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud


# La librairie suivante est nécessaire pour l'affichage graphique de l'arbre:

class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """

    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon
              générique: "att_Numéro")
        """
        self.attribut = num_att  # numéro de l'attribut
        if (nom == ''):  # son nom si connu
            self.nom_attribut = 'att_' + str(num_att)
        else:
            self.nom_attribut = nom
        self.Les_fils = None  # aucun fils à la création, ils seront ajoutés
        self.classe = None  # valeur de la classe si c'est une feuille

    def est_feuille(self):
        """ rend True si l'arbre est une feuille
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None

    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.

    def ajoute_feuille(self, classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe = classe
        self.Les_fils = None  # normalement, pas obligatoire ici, c'est pour être sûr

    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple
            on rend la valeur None si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ', self.nom_attribut, ' -> Valeur inconnue: ', exemple[self.attribut])
            return None

    def compte_feuilles(self):
        """ rend le nombre de feuilles sous ce noeud
        """
        if self.est_feuille():
            return 1
        total = 0
        for noeud in self.Les_fils:
            total += self.Les_fils[noeud].compte_feuilles()
        return total

    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée
        """
        if self.est_feuille():
            g.node(prefixe, str(self.classe), shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i = 0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g, prefixe + str(i))
                g.edge(prefixe, prefixe + str(i), valeur)
                i = i + 1
        return g


class NoeudNumerique:
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """

    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon
              générique: "att_Numéro")
        """
        self.attribut = num_att  # numéro de l'attribut
        if (nom == ''):  # son nom si connu
            self.nom_attribut = 'att_' + str(num_att)
        else:
            self.nom_attribut = nom
        self.seuil = None  #  seuil de coupure pour ce noeud
        self.Les_fils = None  # aucun fils à la création, ils seront ajoutés
        self.classe = None  # valeur de la classe si c'est une feuille

    def est_feuille(self):
        """ rend True si l'arbre est une feuille
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None

    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup

    def ajoute_feuille(self, classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe = classe
        self.Les_fils = None  # normalement, pas obligatoire ici, c'est pour être sûr

    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():  # cas d'arrêt
            return self.classe

        v = exemple[self.attribut]
        if v <= self.seuil:
            return self.Les_fils['inf'].classifie(exemple)
        elif v > self.seuil:
            return self.Les_fils['sup'].classifie(exemple)
        else:
            return 0

    def compte_feuilles(self):
        """ rend le nombre de feuilles sous ce noeud
        """

        if self.est_feuille():
            return 1
        res = 0
        for fils in self.Les_fils:
            res += self.Les_fils[fils].compte_feuilles()

        return res

    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc
            pas expliquée
        """
        if self.est_feuille():
            g.node(prefixe, str(self.classe), shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g, prefixe + "g")
            self.Les_fils['sup'].to_graph(g, prefixe + "d")
            g.edge(prefixe, prefixe + "g", '<=' + str(self.seuil))
            g.edge(prefixe, prefixe + "d", '>' + str(self.seuil))
        return g

def partitionne(m_desc,m_class,n,s):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - n : (int) numéro de colonne de m_desc
            - s : (float) seuil pour le critère d'arrêt
        Hypothèse: m_desc peut être partitionné ! (il contient au moins 2 valeurs différentes)
        output: un tuple composé de 2 tuples
    """
    return ((m_desc[m_desc[:,n]<=s], m_class[m_desc[:,n]<=s]), \
            (m_desc[m_desc[:,n]>s], m_class[m_desc[:,n]>s]))

def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            - nb_classes : (int) nombre initial de labels dans le dataset (défaut: 2)
        output: tuple : ((seuil_trouve, entropie), (liste_coupures,liste_entropies))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropie (float): entropie du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropies (List[float]): la liste des entropies correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropie de la coupure
        val_entropie_inf = entropie(cl_inf) # entropie de l'ensemble des inf
        val_entropie_sup = entropie(cl_sup) # entropie de l'ensemble des sup
        
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    
    return (best_seuil, best_entropie), (liste_coupures,liste_entropies)

def construit_AD_num(X, Y, epsilon, LNoms=[]):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt
        LNoms : liste des noms de features (colonnes) de description
    """

    # dimensions de X:
    (nb_lig, nb_col) = X.shape

    entropie_classe = entropie(Y)

    if (entropie_classe <= epsilon) or (nb_lig <= 1):
        # ARRET : on crée une feuille
        noeud = NoeudNumerique(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = 0.0  #  meilleur gain trouvé (initalisé à 0.0 => aucun gain)
        i_best = -1  #  numéro du meilleur attribut (init à -1 (aucun))

        Xbest_tuple = ()
        Xbest_seuil = ()

        #############

        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_tuple : le tuple rendu par partionne() pour le meilleur attribut trouvé
        # Xbest_seuil : le seuil de partitionnement associé au meilleur attribut
        #
        # Remarque : attention, la fonction discretise() peut renvoyer un tuple contenant
        # None (pas de partitionnement possible)n dans ce cas, on considèrera que le
        # résultat d'un partitionnement est alors ((X,Y),(None,None))

        for i in range(nb_col):  # Pour chaque attribut
            (A, _) = discretise(X, Y, i)
            seuil, entropie_attribut = A
            if seuil is not None:  # Si une discretisation est possible
                gain = entropie_classe - entropie_attribut
                if gain > gain_max:
                    gain_max = gain
                    i_best = i
                    Xbest_seuil = seuil
                    Xbest_tuple = partitionne(X, Y, i, seuil)
            else:
                # Pas de discretisation possible pour cet attribut
                continue

        ############
        if (i_best != -1):  # Un attribut qui amène un gain d'information >0 a été trouvé
            if len(LNoms) > 0:  # si on a des noms de features
                noeud = NoeudNumerique(i_best, LNoms[i_best])
            else:
                noeud = NoeudNumerique(i_best)
            ((left_data, left_class), (right_data, right_class)) = Xbest_tuple
            noeud.ajoute_fils(Xbest_seuil, \
                              construit_AD_num(left_data, left_class, epsilon, LNoms), \
                              construit_AD_num(right_data, right_class, epsilon, LNoms))
        else:  # aucun attribut n'a pu améliorer le gain d'information
            # ARRET : on crée une feuille
            noeud = NoeudNumerique(-1, "Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))

    return noeud

class NoeudNumerique:
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.seuil = None          # seuil de coupure pour ce noeud
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()            
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup        
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.Les_fils['inf'].classifie(exemple)
        return self.Les_fils['sup'].classifie(exemple)

    
    def compte_feuilles(self):
        """ rend le nombre de feuilles sous ce noeud
        """
        if self.est_feuille():
            return 1
        return self.Les_fils['inf'].compte_feuilles() + self.Les_fils['sup'].compte_feuilles()
     
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc 
            pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g,prefixe+"g")
            self.Les_fils['sup'].to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))                
        return g

def echantillonLS(LS,m,avecRemise):
    """ LS: LabeledSet (couple de np.arrays)
        m : entier donnant la taille de l'échantillon voulu (hypothèse: m <= len(LS))
        avecRemise: booléen pour le mode de tirage
    """
    (desc, labels) = LS
    
    # Tirage d'indices
    indices = tirage(range(len(desc)), m, avecRemise)
    
    # Construction de l'échantillon
    desc_ech = desc[indices,:]
    labels_ech = labels[indices]

    return (desc_ech, labels_ech)


class ClassifierBaggingTree(Classifier):
    def __init__(self, input_dimension, b_trees, sample_percent, entropie, avecRemise):
        self.dimension = input_dimension
        self.b_trees = b_trees
        self.sample_percent = sample_percent
        self.entropie = entropie
        self.avecRemise = avecRemise

        self.arbres = []

    def train(self, labeledset):
        # Extraction des données
        desc_set, label_set = labeledset
        self.classes = np.unique(label_set)
        
        # Taille de l'échantillon
        m = int(self.sample_percent * len(desc_set))
        
        # Construction d'un arbre numérique
        for i in range(self.b_trees) :
            desc, labels = echantillonLS(labeledset, m, self.avecRemise)
            arbre = ClassifierArbreNumerique(len(desc_set[0]), self.entropie)
            arbre.train(desc, labels)
            
            self.arbres.append(arbre)

    def score(self, x):
        scores = []
        
        for i in range(self.b_trees):
            scores.append(self.arbres[i].predict(x))
                            
        # Calcul de la proportion de +1 parmi les B arbres
        p = np.sum(np.array(scores) == 1) / self.b_trees
        
        # Calcul du score
        return 2 * (p - 0.5)

    def predict(self, exemple) :
        # Initialisation du dictionnaire de votes
        votes = dict.fromkeys(self.classes, 0)
        
        # Pour chaque arbre, on vote pour la classe prédite
        for arbre in self.arbres :
            votes[arbre.predict(exemple)] += 1
        
        # On retourne la classe ayant le plus de votes
        return max(votes, key=votes.get)
    
class ClassifierMultiOAABaggingTree(Classifier):
    def __init__(self, input_dimension, b_trees, sample_percent, epsilon, avecRemise, labels=[]):
        Classifier.__init__(self, input_dimension)
        
        self.b_trees = b_trees
        self.sample_percent = sample_percent
        self.epsilon = epsilon
        self.avecRemise = avecRemise
        self.labels = labels

        self.modeles = {}

    def train(self, labeledset):
        # Extraction des données
        desc_set, label_set = labeledset
        
        for label in self.labels:
            binary_labels = np.where(label_set == label, 1, -1)
            tree = ClassifierBaggingTree(self.dimension, self.b_trees, self.sample_percent, self.epsilon, self.avecRemise)
            tree.train((desc_set, binary_labels))

            self.modeles[label] = tree

    def score(self, x):
        scores = {}

        for label, modele in self.modeles.items():
            scores[label] = modele.score(x)

        return scores
    
    def predict(self, x):
        scores = self.score(x)
        return max(scores, key=scores.get)


def tirage(VX, m, avecRemise=False):
    """ VX: vecteur d'indices 
        m : nombre d'exemples à sélectionner (hypothèse: m <= len(VX))
        avecRemise: booléen, true si avec remise, ou faux sinon
    """
    # Initialisation de la liste des indices
    indices = []

    # Tirage d'indices
    for i in range(m):
        if avecRemise:
            # Tirage avec remise
            ind = random.choice(VX)
        else:
            # Tirage sans remise
            ind = random.sample(VX, 1)[0]

        indices.append(ind)

    return indices


### Noyau


# Astuce du noyau et projection

#  CLasse (abstraite) pour représenter des noyaux
class Kernel():
    """ Classe pour représenter des fonctions noyau
    """

    def __init__(self, dim_in, dim_out):
        """ Constructeur de Kernel
            Argument:
                - dim_in : dimension de l'espace de départ (entrée du noyau)
                - dim_out: dimension de l'espace de d'arrivée (sortie du noyau)
        """
        self.input_dim = dim_in
        self.output_dim = dim_out

    def get_input_dim(self):
        """ rend la dimension de l'espace de départ
        """
        return self.input_dim

    def get_output_dim(self):
        """ rend la dimension de l'espace d'arrivée
        """
        return self.output_dim

    def transform(self, V):
        """ ndarray -> ndarray
            fonction pour transformer V dans le nouvel espace de représentation
        """
        raise NotImplementedError("Please Implement this method")


class KernelBias(Kernel):
    """ Classe pour un noyau simple 2D -> 3D
    """

    def __init__(self):
        """ Constructeur de KernelBias
            pas d'argument, les dimensions sont figées
        """
        #  Appel du constructeur de la classe mère
        super().__init__(2, 3)

    def transform(self, V):
        """ ndarray de dim 2 -> ndarray de dim 3
            rajoute une 3e dimension au vecteur donné
        """

        if (V.ndim == 1):  #  on regarde si c'est un vecteur ou une matrice
            W = np.array([V])  #  conversion en matrice
            V_proj = np.append(W, np.ones((len(W), 1)), axis=1)

            V_proj = V_proj[0]  #  on rend quelque chose de la même dimension
        else:
            V_proj = np.append(V, np.ones((len(V), 1)), axis=1)

        return V_proj


class KernelPoly(Kernel):
    def __init__(self):
        """ Constructeur de KernelPoly
            pas d'argument, les dimensions sont figées
        """
        #  Appel du constructeur de la classe mère
        super().__init__(2, 6)

    def transform(self, V):
        """ ndarray de dim 2 -> ndarray de dim 6
            ...
        """

        if (V.ndim == 1):  #  on regarde si c'est un vecteur ou une matrice
            W = np.array([V])  # conversion en matrice
            V_proj = np.hstack(([[1]], W, W ** 2, np.prod(W, axis=1).reshape(-1, 1)))

            V_proj = V_proj[0]  #  on rend quelque chose de la même dimension
        else:
            V_proj = np.hstack((np.ones((len(V), 1)), V, V ** 2, np.prod(V, axis=1).reshape(-1, 1)))

        return V_proj

class ClassifierPerceptronKernel(ClassifierPerceptron):
    """ Perceptron de Rosenblatt kernelisé
    """

    def __init__(self, input_dimension, learning_rate, noyau, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : epsilon
                - noyau : Kernel à utiliser
                - init est le mode d'initialisation de w:
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        ClassifierPerceptron.__init__(self, noyau.output_dim, learning_rate, init)
        self.noyau = noyau

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        idx = np.random.permutation(len(label_set))
        Xm = desc_set[idx]
        Ym = label_set[idx]

        for x, y in zip(Xm, Ym):

            p = self.predict(x)

            if p != y:
                ext_x = self.noyau.transform(x)
                self.w += self.learning_rate * y * ext_x

    def score(self, x):
        """ rend le score de prédiction sur x
            x: une description (dans l'espace originel)
        """
        # ext_w = self.noyau.transform(self.w)
        ext_x = self.noyau.transform(x)

        p = self.w @ ext_x

        return p

