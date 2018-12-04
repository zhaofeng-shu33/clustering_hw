#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: zhaofeng-shu33
# license: Apache License V2.0
# file-description: generate Gaussian mixture dataset and use kmeans and Affinity propagation to cluster


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse
from itertools import permutations
import logging
from sihouette_plot import plot_clusters_silhouette 
from sklearn.metrics import silhouette_samples
from sklearn.cluster import AffinityPropagation
import PIL.Image as Image

class data_descriptor(object):
    def __init__(self,data):
        self.data = data
    def plot(self):
        plt.title('data generated from gaussian mixtured model')
        plt.scatter(self.data[:,0], self.data[:,1]) 
        plt.show()        
class gaussian_mixture_generator(data_descriptor):

    def __init__(self, mean, covariance, weight, data_points = 300 ):
        num_component = len(weight)
        mixture_type_list = list(np.random.choice(np.arange(num_component),size=data_points,p=weight))
        whole_data = np.random.multivariate_normal(mean[0,:], covariance[0,:,:], size = mixture_type_list.count(0) )
        for i in range(1,num_component):
            set_tmp = np.random.multivariate_normal(mean[i,:], covariance[i,:,:], size = mixture_type_list.count(i) )
            whole_data = np.concatenate((whole_data,set_tmp))

        super(gaussian_mixture_generator,self).__init__(whole_data)

class image_segementation_data_wrapper(data_descriptor):

    def __init__(self, file_name, use_gradient=False):
        f = Image.open(file_name)
        self.data = np.array(f)
        self.data = self.data[:,:,:3] # remove the alpha channel
        self.image_height = self.data.shape[0]        
        self.image_width = self.data.shape[1]
        self.data =  self.data.reshape(self.data.shape[0]*self.data.shape[1],3) # flatten the data
        

    def plot(self,num_of_clusters, labels,saved_file_name):
        data_tmp = self.data.copy()
        for i in range(num_of_clusters):
            color = np.average(self.data[np.where(labels==i),:],axis=1)
            data_tmp[np.where(labels==i),:] = color
        img = Image.fromarray(data_tmp.reshape(self.image_height,self.image_width,3))            
        img.show()
        img.save(saved_file_name)

class clustering_algorithm(object):
    def __init__(self,input_data):
        self.data = input_data
        self.num = self.data.shape[0] # row, n data points
        self.d = input_data[0,:].shape[0]        
    def plot(self, savefig_name, bench_mark_plot = False):
        color_vector = ['r','b','g','m','y','c','k']
        if (self.k >= 7):
            raise NotImplementedError("plot routine for k >=7 is not implemented")
        if (self.d != 2): # this function can only be used for d = 2
            raise NotImplementedError("plot routine for dimension larger than 2 is not implemented")
        if(bench_mark_plot):
            tmp_mark = self.bench_mark
        else:
            tmp_mark = self.mark
        for i in range(self.k):
            category_i = np.where(tmp_mark==i)[0]
            plt.scatter(self.data[category_i,0],self.data[category_i,1],color=color_vector[i])
       
        plt.savefig(savefig_name)
        plt.show()
    def fit(self):
        '''
        fit the model with self.data
        return the iteration times used
        '''
        self.mark,iteration_cnt = self._implementation()
        return iteration_cnt
        
class affinity_propagation(clustering_algorithm):
    def __init__(self,input_data, init_preference, max_iter = 200):
        self.init_preference = init_preference
        self.max_iteration = max_iter
        super(affinity_propagation,self).__init__(input_data)
    def _implementation(self):
        af = AffinityPropagation(preference=self.init_preference,max_iter=self.max_iteration).fit(self.data)
        self.k = len(af.cluster_centers_indices_)
        return (af.labels_, af.n_iter_)
        
class kmeans(clustering_algorithm):
    def __init__(self,input_data,num_clusters):
        '''
           -----------
           Parameters:
           input_data   : n times d array, with n data points, d  dimension
           num_clusters : the number of clusters to divide the input_data
           
        '''
        self.k = num_clusters
        super(kmeans,self).__init__(input_data)
    def compare(self):
        '''
        -------
        Returns:
        min_mismatch_percent[float]: number of mismatch labels compared with industrial implementation / total label numbers
        inertia_relative_difference[float]: 2*(inertia_1-inertia_2)/(inertia_1+inertia_2)
        '''
        bench_kmeans = KMeans(n_clusters = self.k)
        self.bench_mark = bench_kmeans.fit_predict(self.data)        
       
        # max category label match
        # brute force implementation
        min_mismatch = len(np.where(self.mark!=self.bench_mark)[0])
        best_match_permutation = list(range(self.k))
        for perm in permutations(range(self.k)):
            mark_tmp = self.mark.copy() 
            for j in range(self.num):
                mark_tmp[j]=perm[mark_tmp[j]]
            test_assumption = len(np.where(mark_tmp!=self.bench_mark)[0])
            if(test_assumption < min_mismatch):
                min_mismatch = test_assumption
                best_match_permutation = perm
                
        logging.debug("result compared with bench mark")                
        # first criterion: label mismatch
        logging.debug("label mismatch cnt: %d"%min_mismatch)
        # second criterion: inertia relative difference
        centroid_l2_distance_diff = np.linalg.norm(self.cluster_centers[best_match_permutation,:]-bench_kmeans.cluster_centers_)
        
        # compute inertia: inertia is defined as "Sum of squared distances of samples to their closest cluster center".
        # status : not used
        self.inertia = np.zeros(self.k)
        for i in range(self.k):
            category_i = np.where(self.mark==i)[0]
            self.inertia[i] = np.linalg.norm(self.data[category_i,:]-self.cluster_centers[i,:])     
        total_inertia = np.sum(self.inertia**2)   

        inertia_relative_difference = np.abs(total_inertia - bench_kmeans.inertia_)/np.mean([total_inertia,bench_kmeans.inertia_])
        logging.debug("inertia relative difference: %f"%(inertia_relative_difference))
        return (min_mismatch/self.num,inertia_relative_difference)

    def _implementation(self, tolerance = 1e-3):
        # we take initial centroid to be the mean of randomly sampled points
        m  = np.zeros([self.k,self.d])        
        m  = self.data[np.random.choice(np.arange(self.num),size=self.k,replace=False),:]
        # k kinds of S sets are implemented by k boolean arrays
        S = np.zeros(self.num, dtype=int)
        last_m = np.zeros([self.k,self.d])
        iteration_cnt = 0
        while(np.linalg.norm(m-last_m) >= tolerance and iteration_cnt < 200): # 200 is maximum iteration count
            debug_error_residue = np.linalg.norm(m-last_m)
            logging.debug('debug, cnt: {0} center oscilation L2 norm {1}'.format(iteration_cnt,debug_error_residue))           
            last_m = m.copy()
            #***********Assignment***************
            for i in range(self.num):
                S[i] = np.argmin(np.linalg.norm(self.data[i,:]-m,axis=1))
            #***********Update******************
            for i in range(self.k):
                index_list = np.where(S==i)[0]
                if not(len(index_list)==0):                    
                    m[i,:]=np.mean(self.data[index_list,:],axis=0)
            iteration_cnt += 1
        self.cluster_centers = m
        logging.debug('debug, cnt: {0} center oscilation L2 norm{1}'.format(iteration_cnt,np.linalg.norm(m-last_m)))
        
        return (S,iteration_cnt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare",dest='compare_time', help="compare self-implementation with sklearn implementation", type=int)
    parser.add_argument("--plot", dest='plot_file_name', help="plot the clustering result, save the plot to the user given file name")
    parser.add_argument("--sihouette", dest='plot_sihouette_name', help="plot the sihouette, save the plot to the user given file name")
    parser.add_argument("--debug", help="debug the program, logger can output debug info",action="store_true")    
    parser.add_argument("--uneven", help="make the data anisotropicly distributed",action="store_true")    
    parser.add_argument("--method", help="cluster algorithm to use,default to k-means", choices=['k-means', 'affinity_propagation'], default="k-means")
    parser.add_argument("--num_of_clusters", help="num of clusters for k-means algorithm", type=int)    
    parser.add_argument("--image", dest='image_file_name', help="use given image file to do segementation task with clustering algorithm")    
    args = parser.parse_args()    
    if(args.debug):
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # gaussian-mixture model(dynamic generating)
    if(args.uneven):
        mu_1 = np.array([1,1])
        mu_2 = np.array([5,1])
        mu = np.concatenate((mu_1,mu_2)).reshape(2,2)
        # covariance matrix
        C_1 = np.array([[0.5,0],[0,8]])
        C_2 = np.array([[0.5,0],[0,8]])
        Cov = np.concatenate((C_1,C_2)).reshape(2,2,2)
        
        # mixture weights
        w_1 = 0.5
        w_2 = 1 - w_1
        weight = [w_1,w_2]
        heuristic_num_clusters = 2
        num_of_samples = 1500
    else:
        # mean
        mu_1 = np.array([10,3])
        mu_2 = np.array([1,1])
        mu_3 = np.array([5,4])
        mu = np.concatenate((mu_1,mu_2,mu_3)).reshape(3,2)
        # covariance matrix
        C_1 = np.array([[1,0],[0,1]])
        C_2 = np.array([[1.5,0],[0,1.5]])
        C_3 = np.array([[2,0],[0,2]])
        Cov = np.concatenate((C_1,C_2,C_3)).reshape(3,2,2)
        
        # mixture weights
        w_1 = 0.33
        w_2 = 0.33
        w_3 = 1 - w_1 - w_2
        weight = [w_1,w_2,w_3]
        heuristic_num_clusters = 3
        num_of_samples = 300
    if(args.num_of_clusters):
        heuristic_num_clusters = args.num_of_clusters
                
    if(args.image_file_name):
        dg_instance = image_segementation_data_wrapper(args.image_file_name)
    else:
        dg_instance = gaussian_mixture_generator(mu, Cov, weight, data_points=num_of_samples)
    if(args.method == 'k-means'):
        kmeans_instance_from_dg = kmeans(dg_instance.data, num_clusters=heuristic_num_clusters)    
        clustering_instance = kmeans_instance_from_dg
    elif(args.method == 'affinity_propagation'):
        if(args.uneven):
            preference_fine_tuning = -600
            max_iteration = 1000 # still possible k>=7, which broke the plot routine out.
        else:
            preference_fine_tuning = -250                
            max_iteration = 200
        af_from_dg = affinity_propagation(dg_instance.data, init_preference = preference_fine_tuning, max_iter=max_iteration) # the default iteration is not enough!
        clustering_instance = af_from_dg
    else:
        raise NotImplementedError("method %s is not implemented!"%args.method)
    clustering_instance.fit()

        
    if(args.compare_time):
        sum_min_mismatch_percent = 0
        sum_inertia_relative_difference = 0
        sum_iteration_cnt = 0
        for i in range(args.compare_time):
            dg_instance = gaussian_mixture_generator(mu, Cov, weight, data_points=num_of_samples)
            clustering_instance = kmeans(dg_instance.data, num_clusters=heuristic_num_clusters)    
            iteration_cnt = clustering_instance.fit()
            min_mismatch_percent, inertia_relative_difference= clustering_instance.compare()
            
            sum_min_mismatch_percent    += min_mismatch_percent
            sum_inertia_relative_difference += inertia_relative_difference
            sum_iteration_cnt += iteration_cnt
        logging.info("average mismatch : %f "%(sum_min_mismatch_percent*1.0/args.compare_time))
        logging.info("average inertia relative difference : %f" %(sum_inertia_relative_difference/args.compare_time))
        logging.info("average iteration times: %f"%(sum_iteration_cnt*1.0/args.compare_time))
    if(args.plot_file_name):
        if(args.image_file_name):
            dg_instance.plot(heuristic_num_clusters, clustering_instance.mark, args.plot_file_name)
        else:
            dg_instance.plot()
            clustering_instance.plot(args.plot_file_name)
    if(args.plot_sihouette_name):
        sample_silhouette_values = silhouette_samples(clustering_instance.data,clustering_instance.mark)
        logging.info("sihouette_score: %f" % np.mean(sample_silhouette_values))
        plot_clusters_silhouette(sample_silhouette_values, clustering_instance.mark, heuristic_num_clusters, args.plot_sihouette_name)
