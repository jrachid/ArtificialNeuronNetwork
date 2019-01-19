# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 09:02:37 2018

@author: Rachid
"""

# Data preprocessing

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importer le dataset 
dataset = pd.read_csv("Churn_Modelling.csv")

# le clustering sera fait avec le "annual income" et "le spending score"
# difficile d'identifier des clusters avec l'âge et plus facile de visualiser sur 2 dimensions
X = dataset.iloc[:,[6,12]].values

# Pas de Feature scaling, car les variables sont à la même échelle (de 0 à 100)

# utiliser le modèle Elbow pour trouver le nombre optimal de cluster
from sklearn.cluster import KMeans

# création et intialisation de la liste WCSS
wcss = []

# on calcule dix valeur de WCSS pour obtenir un graphique et déduire le nombre de clusters optimal
for i in range(1,11):
    # créer l'objet de la classe Kmeans
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    # on lie le kmeans avec les variables dépendantes
    kmeans.fit(X)
    # on récupère le wcss et on l'ajoute à la liste
    wcss.append(kmeans.inertia_)
    
# visualisation du graphe elbow
plt.plot(range(1,11), wcss)
plt.title("Elbow method")
plt.xlabel("Nombre de clusters")
plt.ylabel("WCSS")
plt.show()
    
# A la visualation du graphe on peut en conclure que le nombre de cluster optimal est K=5

#♣ Construction du modèle
# je recrée un objet kmeans mais en spécifiant le nombre de cluster obtenu précedemment
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)

# lier Kmeans aux variables indépendantes du dataset X et obtenir les clusters avec
# la fonction predict
# on obtiendra dans y_kmeans le cluster auquel appartient chaque point d'observation
y_kmeans = kmeans.fit_predict(X)

# Représentation sur un graphe des différents clusters et les points d'observations
# placement des points d'observations et par couleur de cluster sur le graphe
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], c = 'red', label = "Cluster 1" )
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c = 'blue', label = "Cluster 2" )
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], c = 'green', label = "Cluster 3" )
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], c = 'cyan', label = "Cluster 4" )
# plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], c = 'magenta', label = "Cluster 5" )
plt.title("Clustering")
plt.xlabel("Credit Score")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()