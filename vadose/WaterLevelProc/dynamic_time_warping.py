# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:33:25 2018

@author: amoody
"""
from sklearn.metrics import classification_report
import random
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

def DTWDistance(s1, s2,w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return sqrt(LB_sum)



def knn(train,test,w):
    preds=[]
    for ind,i in enumerate(test):
        min_dist=float('inf')
        closest_seq=[]
        #print ind
        for j in train:
            if LB_Keogh(i[:-1],j[:-1],5)<min_dist:
                dist=DTWDistance(i[:-1],j[:-1],w)
               
                if (dist<min_dist) & (dist != 0.):                 
                    f,a = plt.subplots(1,1)
                    a.plot(i[:-1])
                    a.plot(j[:-1])
                    a.set_title('DTWD = {}'.format(dist))
                    min_dist=dist
                    closest_seq=j
        preds.append(closest_seq[-1])
    return classification_report(test[:,-1],preds)

def k_means_clust(data,num_clust,num_iter,w=5):
    centroids=random.sample(list(data),num_clust)
    counter=0
    for n in range(num_iter):
        counter+=1
        print(counter)
        assignments={}
        #assign data points to clusters
        for ind,i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                if LB_Keogh(i,j,5)<min_dist:
                    cur_dist=DTWDistance(i,j,w)
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust]=[]

        #recalculate centroids of clusters
        for key in assignments:
            clust_sum=0
            for k in assignments[key]:
                clust_sum=clust_sum+data[k]
            centroids[key]=[m/len(assignments[key]) for m in clust_sum]

    return centroids

def skclust( data,n_clust ):
    from cycler import cycler
    f,ax = plt.subplots(1,3,figsize=(10,4))
    knn_graph = kneighbors_graph(data,10,include_self=False)
    d=dict(zip(range(0,6),['k','c','m','y','r','g']))
    for i, linkage in enumerate(('average','complete','ward')):
        print(linkage)
        model = AgglomerativeClustering(linkage = linkage,
                                        connectivity= knn_graph,
                                        n_clusters= n_clust)
        model.fit(data)
        plt.rc('axes',prop_cycle=cycler('color',[d[l] for l in model.labels_]))
        ax[i].plot(data.T)
        ax[i].set_title('linkage=%s'%(linkage),fontsize=9)
        
    plt.rc(plt.rcParamsDefault)
    return
        
        
    
    
# These functions work on rows as individual time series!
#knn(train.T,test.T,4)

centroids = k_means_clust(data_ma.values.T,5,20)
skclust(data_ma.values.T,5)
