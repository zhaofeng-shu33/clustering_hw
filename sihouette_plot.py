"""
Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).

Modified: zhaofeng-shu33
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import os
import logging
import pdb
def plot_clusters_silhouette(sample_silhouette_values, cluster_labels, n_clusters,
                             file_name):
    """Plot the silhouette score for each cluster, given the distance matrix X.
    Parameters
    ----------
    sample_silhouette_values : array_like, silhouette scores of data
    cluster_labels : array_like
        List of integers which represents the cluster of the corresponding
        point . The size must be the same has a dimension of sample_silhouette_values.
    n_clusters : int
        The number of clusters.
    file_name : file name for output images.
    """
    # Create a subplot with 1 row and 2 columns
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(20, 15)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    # ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(sample_silhouette_values) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters

    # Compute the silhouette scores for each sample
    # sample_silhouette_values is numpy array
 
    silhouette_avg = np.mean(sample_silhouette_values)
    logging.info("Average silhouette_score: %.4f", silhouette_avg)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        # ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title(("Silhouette analysis (n_clusters {}, avg score {:.4f}, "
                  "tot samples {}".format(n_clusters, silhouette_avg, len(sample_silhouette_values))),
                 fontsize=24, fontweight='bold')
    ax1.set_xlabel("silhouette coefficient values",fontsize=20)
    ax1.set_ylabel("cluster label",fontsize=20)

    # The vertical line for average silhoutte score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.tick_params(axis='both', which='major', labelsize=20)
    ax1.set_yticks([])  # Clear the yaxis labels / ticks    
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    #pdb.set_trace()
    fig.savefig(file_name)
    plt.show()
    logging.info('Figured saved %s', file_name)