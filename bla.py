import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.metrics import pairwise_distances

# Load the dataset and select relevant columns
df=pd.read_csv('Customer Dataset- clustering.csv')
cdf=df[['BALANCE','BALANCE_FREQUENCY','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY','CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE']]
cdf = cdf.dropna()

##define a function to calculate the disturbance matrix
def calculate_disturbance_matrix(x, labels):
    distances = pairwise_distances(x)
    mask = labels[:, np.newaxis] == labels
    disturbance_matrix = np.where(mask, distances, 0)
    return disturbance_matrix

##define a function to evaluate the clustering algotirhm based on the disturbance matrix
def evaluate_clustering(disturbance_matrix):
    return np.sum(disturbance_matrix)/2

##compare different clustering algorithms and find the optimal number of clusters
num_cluster_range=range(2,6)


# Initialize empty lists to store the number of clusters and evaluations for each clustering algorithm
num_clusters_list = []
kmeans_evaluations = []
agglomerative_evaluations = []
dbscan_evaluations = []

for num_clusters in num_cluster_range:
    # Initialize clustering algorithms with the current number of clusters
    kmeans=KMeans(n_clusters=num_clusters, n_init=10)
    Agglomerative=AgglomerativeClustering(n_clusters=num_clusters)
    dbscan=DBSCAN(eps=0.5 ,min_samples=5 )##default numbers


    # Fit and predict labels for each algorithm
    kmeans_labels=kmeans.fit_predict(cdf)
    Agglomerative_labels=Agglomerative.fit_predict(cdf)
    dbscan_labels=dbscan.fit_predict(cdf)

    # Calculate disturbance matrix for each algorithm
    kmeans_disturbance_matrix=calculate_disturbance_matrix(cdf,kmeans_labels)
    Agglomerative_disturbance_matrix=calculate_disturbance_matrix(cdf,Agglomerative_labels)
    dbscan_disturbance_matrix=calculate_disturbance_matrix(cdf,dbscan_labels)

    # Evaluate each algorithm using the disturbance matrix
    kmeans_evaluation=evaluate_clustering(kmeans_disturbance_matrix)
    Agglomerative_evaluation=evaluate_clustering(Agglomerative_disturbance_matrix)
    dbscan_evaluation=evaluate_clustering(dbscan_disturbance_matrix)

    # Store evaluation results for each algorithm in lists
    num_clusters_list.append(num_clusters)
    kmeans_evaluations.append(kmeans_evaluation)
    agglomerative_evaluations.append(Agglomerative_evaluation)
    dbscan_evaluations.append(dbscan_evaluation)


# Create a figure and 3 subplots with a size of 6x12 inches
fig, axs = plt.subplots(3, figsize=(6, 12))

# Plot the k-means evaluations on the first subplot
axs[0].plot(num_clusters_list, kmeans_evaluations)
axs[0].set_title('K-means')
axs[0].set_xlabel('Number of clusters')
axs[0].set_ylabel('Evaluation')

# Plot the agglomerative evaluations on the second subplot
axs[1].plot(num_clusters_list, agglomerative_evaluations)
axs[1].set_title('Agglomerative')
axs[1].set_xlabel('Number of clusters')
axs[1].set_ylabel('Evaluation')

# Plot the DBSCAN evaluations on the third subplot
axs[2].plot(num_clusters_list, dbscan_evaluations)
axs[2].set_title('DBSCAN')
axs[2].set_xlabel('Number of clusters')
axs[2].set_ylabel('Evaluation')

# Adjust the layout of the subplots to fit in the figure
plt.tight_layout()
plt.show()