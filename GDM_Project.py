#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:41:23 2020

@author: sohailnizam
"""

test = np.load("./mb8_test.npy")
test_abs = np.absolute(test)
np.fill_diagonal(test_abs, 0)

test_graph = from_numpy_array(test_abs)
test_aw = AnonymousWalks(test_graph)
test_embedding, test_meta = test_aw.embed(steps = 3, method = 'sampling', 
                                keep_last=True, verbose=False, MC = 100)

print(test_embedding)


test2 = np.load("./graphs/sb2/sb_4.npy")
test2_abs = np.absolute(test2)
np.fill_diagonal(test2_abs, 0)

test2_graph = from_numpy_array(test2_abs)
test2_aw = AnonymousWalks(test2_graph)
test2_embedding, test2_meta = test2_aw.embed(steps = 3, method = 'sampling', 
                                keep_last=True, verbose=False, MC = 100)

print(test_embedding)

### Get raw correlation matrix data ###

path = './AWE'
os.chdir(path)
from AnonymousWalkKernel import GraphKernel
from AnonymousWalkKernel import AnonymousWalks
from AnonymousWalkKernel import Evaluation
import numpy as np
import networkx as nx
from networkx.convert_matrix import from_numpy_array, to_numpy_matrix
import time
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


'''
We want all 9*32 graphs.
9 mbs, 32 subjects per mb

'''
path = "./Desktop/graphs/"
def import_mb_mats(mb):
    
    '''
    Take in a str representing mb of interest (e.g 'mb4')
    For load each of the 32 graphs for that mb as numpy mats
    Get their absolute values
    Set their diagonals to 0
    Append them to a list
    Return that list of numpy mats
    
    '''
    
    mat_list = []
    for i in range(1, 33):
        mat = np.load(path + mb + '/' + mb + '_' + str(i) + '.npy')
        mat = np.absolute(mat)
        np.fill_diagonal(mat, 0)
        mat_list.append(mat)
        
    return(mat_list)
    
sb3_mat_list = import_mb_mats('sb3')
sb2_mat_list = import_mb_mats('sb2')
mb2_mat_list = import_mb_mats('mb2')
mb3_mat_list = import_mb_mats('mb3')
mb4_mat_list = import_mb_mats('mb4')
mb6_mat_list = import_mb_mats('mb6')
mb8_mat_list = import_mb_mats('mb8')
mb9_mat_list = import_mb_mats('mb9')
mb12_mat_list = import_mb_mats('mb12')


### Stage 1: Correlation Matrix Analysis ###

## 1a. Thresholded corrs at various walk sizes for select subjects ##

#Let's pick 3 subjects randomly: 4, 20, 31
#and get their AWEs at k = 3, 5, and 10 for each mb


#A fcn to take a list of np mats
#and return a list of embeddings
def get_embeddings(mat_list, k, thresh = None, MC = 10000):
    
    
    embedding_list = []
    count = 0
    
    for sub in mat_list:
        
        #if the max val <= thresh, we'll have all 0s
        #so move on to the next matrix
        if thresh is not None  and np.max(sub) <= thresh:
            continue
        
        if thresh is not None:
            sub = np.where(sub > thresh, 1, 0)
        
        #cast to nx graph
        graph = from_numpy_array(sub)
        aw = AnonymousWalks(graph)
        
        embedding, meta = aw.embed(steps = k, method = 'sampling', 
                        keep_last=True, verbose=False, MC = MC)
        
        print(meta)
        
        embedding_list.append(embedding)
        
        count += 1
        
        print(count, 'done.')
    
    return(embedding_list)
        
   
test = mb9_mat_list[19]
test = np.where(test > .7, 1, 0)



#Subject 4
sub4 = [sb3_mat_list[3], sb2_mat_list[3], mb2_mat_list[3],
        mb3_mat_list[3], mb4_mat_list[3], mb6_mat_list[3],
        mb8_mat_list[3], mb9_mat_list[3], mb12_mat_list[3]]

#Subject 20
sub20 = [sb3_mat_list[19], sb2_mat_list[19], mb2_mat_list[19],
        mb3_mat_list[19], mb4_mat_list[19], mb6_mat_list[19],
        mb8_mat_list[19], mb9_mat_list[19], mb12_mat_list[19]]


#Subject 31
sub31 = [sb3_mat_list[20], sb2_mat_list[30], mb2_mat_list[30],
        mb3_mat_list[30], mb4_mat_list[30], mb6_mat_list[30],
        mb8_mat_list[30], mb9_mat_list[30], mb12_mat_list[30]]


#a fcn that takes a list of mats
#returns a dict storing embeddings for each given walk size
#and each given threshold size (including None) in a dict        
def get_thresh_embeds(sub_list, k_list, thresh_list):
    embed_dict = {}
    for k in k_list:
        embed_dict[k] = {}
        for thresh in thresh_list:
            embed_dict[k][thresh] = get_embeddings(sub_list, k = k, thresh = thresh)
    return(embed_dict)


    

t0 = time.time()
sub4_embed_dict = get_thresh_embeds(sub4, [3, 5, 10], [None, .1, .3, .5, .6])
sub20_embed_dict = get_thresh_embeds(sub20, [3, 5, 10], [None, .1, .3, .5, .6])
sub31_embed_dict = get_thresh_embeds(sub31, [3, 5, 10], [None, .1, .3, .5, .6])         
print('Done in', (time.time() - t0) / 60, 'minutes.')


sns.barplot(x="x", y="y", 
            data=pd.DataFrame({'x' : list(range(0, 52)), 'y' : sub20_embed_dict[5][.5][0]}))


sns.barplot(x="x", y="y", 
            data=pd.DataFrame({'x' : list(range(0, 52)), 'y' : sub20_embed_dict[5][.5][1]}))

sns.barplot(x="x", y="y", 
            data=pd.DataFrame({'x' : list(range(0, 52)), 'y' : sub20_embed_dict[5][.5][2]}))

sns.barplot(x="x", y="y", 
            data=pd.DataFrame({'x' : list(range(0, 52)), 'y' : sub20_embed_dict[5][.5][3]}))

sns.barplot(x="x", y="y", 
            data=pd.DataFrame({'x' : list(range(0, 52)), 'y' : sub20_embed_dict[5][.5][4]}))

sns.barplot(x="x", y="y", 
            data=pd.DataFrame({'x' : list(range(0, 52)), 'y' : sub20_embed_dict[5][.5][5]}))

sns.barplot(x="x", y="y", 
            data=pd.DataFrame({'x' : list(range(0, 52)), 'y' : sub20_embed_dict[5][.5][6]}))

sns.barplot(x="x", y="y", 
            data=pd.DataFrame({'x' : list(range(0, 52)), 'y' : sub20_embed_dict[5][.5][7]}))

sns.barplot(x="x", y="y", 
            data=pd.DataFrame({'x' : list(range(0, 52)), 'y' : sub20_embed_dict[5][.5][8]}))

    


## 1b. Thresholded mean corr mats at various walk sizes ##
path = './Desktop/graphs/mean_corr/'
mean_corr = []
mb_list = ['sb3', 'sb2', 'mb2', 'mb3', 'mb4', 'mb6', 'mb8', 'mb9', 'mb12']
for mb in mb_list:
    mat = np.load(path + mb + '_mc' + '.npy')
    mat = np.absolute(mat)
    np.fill_diagonal(mat, 0)
    mean_corr.append(mat)

mean_corr_embed_dict = get_thresh_embeds(mean_corr, [3, 5, 10], [None, .1, .3, .5, .6])

#create several barplots
sns.barplot(x="Walk", y="Proportion", 
            data=pd.DataFrame({'Walk' : list(range(0, 5)), 'Proportion' : mean_corr_embed_dict[3][None][0]})).set_title("SB3 AWE Distribution, No Threshold")

sns.barplot(x="Walk", y="Proportion", 
            data=pd.DataFrame({'Walk' : list(range(0, 5)), 'Proportion' : mean_corr_embed_dict[3][None][8]})).set_title("MB12 AWE Distribution, No Threshold")


sns.barplot(x="Walk", y="Proportion", 
            data=pd.DataFrame({'Walk' : list(range(0, 5)), 'Proportion' : mean_corr_embed_dict[3][.3][0]})).set_title("SB3 AWE Distribution, T = .3")

sns.barplot(x="Walk", y="Proportion", 
            data=pd.DataFrame({'Walk' : list(range(0, 5)), 'Proportion' : mean_corr_embed_dict[3][.3][8]})).set_title("MB12 AWE Distribution, T = .3")



sns.barplot(x="Walk", y="Proportion", 
            data=pd.DataFrame({'Walk' : list(range(0, 5)), 'Proportion' : mean_corr_embed_dict[3][.5][0]})).set_title("SB3 AWE Distribution, T = .5")

sns.barplot(x="Walk", y="Proportion", 
            data=pd.DataFrame({'Walk' : list(range(0, 5)), 'Proportion' : mean_corr_embed_dict[3][.5][8]})).set_title("MB12 AWE Distribution, T = .5")


# summary measures
apl_list_t4= []
for mb in mean_corr:
    mat = np.where(mb <= .1, 0 , 1)
    mat = from_numpy_array(mat)
    apl = nx.average_shortest_path_length(mat)
    apl_list_t3.append(apl)
    #cc = round(nx.average_clustering(mat, weight='weight'), 4)
    #print(apl, cc)
    #degrees = np.array(list(dict(mat.degree()).values()))
    #mean = np.mean(degrees)
    #deg_list_t11.append(mean)
    #sd = round(np.std(degrees), 2)
    #mini = round(np.min(degrees), 2)
    #q1 = round(np.quantile(degrees, .25), 2)
    #q2 = round(np.quantile(degrees, .5), 2)
    #q3 = round(np.quantile(degrees, .75), 2)
    #maxi = round(np.max(degrees), 2)
    #print(mean, sd, mini, q1, q2, q3, maxi)

deg_df = np.array([deg_list_t1, deg_list_t2, deg_list_t3,
                  deg_list_t4, deg_list_t5, deg_list_t6,
                  deg_list_t7, deg_list_t8, deg_list_t9, 
                  deg_list_t10, deg_list_t11])

deg_df = pd.DataFrame(deg_df) 
deg_df.to_csv('.AWE/degree_df.csv', index = False)

## 1c. Thresholded corr mat AWE PCA visualization and SVM ##

#First want a matrix of embeddings with no thresholds
#that's 32*9 rows (one row per embedding/matrix)
#and 52 columns (one col per walk type)
#we'll do this one mb at a time and then stack
get_embeddings([sb3_mat_list[0]], k = 5, thresh = None, MC = 1000)
sb3_embeds = get_embeddings(sb3_mat_list, k = 5, thresh = None, MC = 1000)
sb2_embeds = get_embeddings(sb2_mat_list, k = 5, thresh = None, MC = 1000)
mb2_embeds = get_embeddings(mb2_mat_list, k = 5, thresh = None, MC = 1000)
mb3_embeds = get_embeddings(mb3_mat_list, k = 5, thresh = None, MC = 1000)
mb4_embeds = get_embeddings(mb4_mat_list, k = 5, thresh = None, MC = 1000)
mb6_embeds = get_embeddings(mb6_mat_list, k = 5, thresh = None, MC = 1000)
mb8_embeds = get_embeddings(mb8_mat_list, k = 5, thresh = None, MC = 1000)
mb9_embeds = get_embeddings(mb9_mat_list, k = 5, thresh = None, MC = 1000)
mb12_embeds = get_embeddings(mb12_mat_list, k = 5, thresh = None, MC = 1000)

embed_tup = (np.array(sb3_embeds), np.array(sb2_embeds), np.array(mb2_embeds),
             np.array(mb3_embeds), np.array(mb4_embeds), np.array(mb6_embeds),
             np.array(mb8_embeds), np.array(mb9_embeds), np.array(mb12_embeds))

embed_mat1 = np.vstack(embed_tup)

#create the labels
label_vec = []
for i in range(1,10):
    label_vec += [i]*32
#label_vec
label_vec = np.array(label_vec).reshape(-1, 1)

#PCA
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
pca = PCA(n_components=2)
components = pca.fit_transform(embed_mat1)
print(pca.explained_variance_ratio_)
plt.scatter(components[:, 0], components[:, 1],
c=label_vec, edgecolor='none')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('AWE Visualization, No Threhsold')

##SVM

#extract just sb3 and mb12
X1 = embed_mat1[0:32, :]
X2 = embed_mat1[256:, :]
X = np.vstack((X1, X2))
y = np.array([0]*32 + [1]*32)
svc = svm.SVC()
print(np.mean(cross_val_score(svc, X, y, cv=5)))

#now extract just mb2 and mb9
X1 = embed_mat1[64:96, :]
X2 = embed_mat1[224: 256, :]
X = np.vstack((X1, X2))
y = np.array([0]*32 + [1]*32)
svc = svm.SVC()
print(np.mean(cross_val_score(svc, X, y, cv=5)))

#now extract just mb4 and mb8
X1 = embed_mat1[128:160, :]
X2 = embed_mat1[192: 224, :]
X = np.vstack((X1, X2))
y = np.array([0]*32 + [1]*32)
svc = svm.SVC()
print(np.mean(cross_val_score(svc, X, y, cv=5)))

#Now the whole thing over again for threshold = .5

sb3_embeds2 = get_embeddings(sb3_mat_list, k = 5, thresh = .5, MC = 1000)
sb2_embeds2 = get_embeddings(sb2_mat_list, k = 5, thresh = .5, MC = 1000)
mb2_embeds2 = get_embeddings(mb2_mat_list, k = 5, thresh = .5, MC = 1000)
mb3_embeds2 = get_embeddings(mb3_mat_list, k = 5, thresh = .5, MC = 1000)
mb4_embeds2 = get_embeddings(mb4_mat_list, k = 5, thresh = .5, MC = 1000)
mb6_embeds2 = get_embeddings(mb6_mat_list, k = 5, thresh = .5, MC = 1000)
mb8_embeds2 = get_embeddings(mb8_mat_list, k = 5, thresh = .5, MC = 1000)
mb9_embeds2 = get_embeddings(mb9_mat_list, k = 5, thresh = .5, MC = 1000)
mb12_embeds2 = get_embeddings(mb12_mat_list, k = 5, thresh = .5, MC = 1000)


embed_tup2 = (np.array(sb3_embeds2), np.array(sb2_embeds2), np.array(mb2_embeds2),
             np.array(mb3_embeds2), np.array(mb4_embeds2), np.array(mb6_embeds2),
             np.array(mb8_embeds2), np.array(mb9_embeds2), np.array(mb12_embeds2))

embed_mat2 = np.vstack(embed_tup2)

pca = PCA(n_components=2)
components = pca.fit_transform(embed_mat2)
print(pca.explained_variance_ratio_)
plt.scatter(components[:, 0], components[:, 1],
c=label_vec, edgecolor='none')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('AWE Visualization, K = 5, Thresold at .5')

##SVM

#extract just sb3 and mb12
X1 = embed_mat2[0:32, :]
X2 = embed_mat2[256:, :]
X = np.vstack((X1, X2))
y = np.array([0]*32 + [1]*32)
svc = svm.SVC()
print(np.mean(cross_val_score(svc, X, y, cv=5)))

#now extract just mb2 and mb9
X1 = embed_mat2[64:96, :]
X2 = embed_mat2[224: 256, :]
X = np.vstack((X1, X2))
y = np.array([0]*32 + [1]*32)
svc = svm.SVC()
print(np.mean(cross_val_score(svc, X, y, cv=5)))

#now extract just mb4 and mb8
X1 = embed_mat2[128:160, :]
X2 = embed_mat2[192: 224, :]
X = np.vstack((X1, X2))
y = np.array([0]*32 + [1]*32)
svc = svm.SVC()
print(np.mean(cross_val_score(svc, X, y, cv=5)))


#now the whole thing over again with k = 7
sb3_embeds3 = get_embeddings(sb3_mat_list, k = 7, thresh = .5, MC = 1000)
sb2_embeds3 = get_embeddings(sb2_mat_list, k = 7, thresh = .5, MC = 1000)
mb2_embeds3 = get_embeddings(mb2_mat_list, k = 7, thresh = .5, MC = 1000)
mb3_embeds3 = get_embeddings(mb3_mat_list, k = 7, thresh = .5, MC = 1000)
mb4_embeds3 = get_embeddings(mb4_mat_list, k = 7, thresh = .5, MC = 1000)
mb6_embeds3 = get_embeddings(mb6_mat_list, k = 7, thresh = .5, MC = 1000)
mb8_embeds3 = get_embeddings(mb8_mat_list, k = 7, thresh = .5, MC = 1000)
mb9_embeds3 = get_embeddings(mb9_mat_list, k = 7, thresh = .5, MC = 1000)
mb12_embeds3 = get_embeddings(mb12_mat_list, k = 7, thresh = .5, MC = 1000)


embed_tup3 = (np.array(sb3_embeds3), np.array(sb2_embeds3), np.array(mb2_embeds3),
             np.array(mb3_embeds3), np.array(mb4_embeds3), np.array(mb6_embeds3),
             np.array(mb8_embeds3), np.array(mb9_embeds3), np.array(mb12_embeds3))

embed_mat3 = np.vstack(embed_tup3)

pca = PCA(n_components=2)
components = pca.fit_transform(embed_mat3)
print(pca.explained_variance_ratio_)
plt.scatter(components[:, 0], components[:, 1],
c=label_vec, edgecolor='none')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('AWE Visualization, K = 7, Thresold at .5')



##SVM

#extract just sb3 and mb12
X1 = embed_mat3[0:32, :]
X2 = embed_mat3[256:, :]
X = np.vstack((X1, X2))
y = np.array([0]*32 + [1]*32)
svc = svm.SVC()
print(np.mean(cross_val_score(svc, X, y, cv=5)))

#now extract just mb2 and mb9
X1 = embed_mat3[64:96, :]
X2 = embed_mat3[224: 256, :]
X = np.vstack((X1, X2))
y = np.array([0]*32 + [1]*32)
svc = svm.SVC()
print(np.mean(cross_val_score(svc, X, y, cv=5)))

#now extract just mb4 and mb8
X1 = embed_mat3[128:160, :]
X2 = embed_mat3[192: 224, :]
X = np.vstack((X1, X2))
y = np.array([0]*32 + [1]*32)
svc = svm.SVC()
print(np.mean(cross_val_score(svc, X, y, cv=5)))


###############

def awe_summary(embed_list):
    highest_dict = {}
    sec_highest_dict = {}
    third_highest_dict = {}
    
    
    for embed in embed_list:
        if embed.index(max(embed)) not in highest_dict.keys():
            highest_dict[embed.index(max(embed))] = 1
        else:
            highest_dict[embed.index(max(embed))] += 1
            
        if embed.index(sorted(embed)[-2]) not in sec_highest_dict.keys():
            sec_highest_dict[embed.index(sorted(embed)[-2])] = 1
        else:
            sec_highest_dict[embed.index(sorted(embed)[-2])] += 1
            
        if embed.index(sorted(embed)[-3]) not in third_highest_dict.keys():
            third_highest_dict[embed.index(sorted(embed)[-3])] = 1
        else:
            third_highest_dict[embed.index(sorted(embed)[-3])] += 1
            
            
    return(highest_dict, sec_highest_dict, third_highest_dict)
            

awe_summary(sb3_embeds2)
awe_summary(sb2_embeds2)
awe_summary(mb2_embeds2)
awe_summary(mb3_embeds2)
awe_summary(mb4_embeds2)
awe_summary(mb6_embeds2)
awe_summary(mb8_embeds2)
awe_summary(mb9_embeds2)
awe_summary(mb12_embeds2)

five_walks = [[0, 1, 2, 0, 1, 0], [0, 1, 0, 2, 1, 0], [0, 1, 2, 3, 1, 0],
                [0, 1, 2, 0, 2, 0], [0, 1, 0, 1, 2, 0], [0, 1, 2, 1, 2, 0], 
                [0, 1, 2, 3, 2, 0], [0, 1, 2, 0, 3, 0], [0, 1, 2, 1, 3, 0],
                [0, 1, 0, 2, 3, 0], [0, 1, 2, 3, 4, 0], [0, 1, 0, 1, 0, 1],
                [0, 1, 2, 1, 0, 1], [0, 1, 0, 2, 0, 1], [0, 1, 2, 3, 0, 1],
                [0, 1, 2, 0, 2, 1], [0, 1, 0, 1, 2, 1], [0, 1, 2, 1, 2, 1], 
                [0, 1, 2, 3, 2, 1], [0, 1, 2, 0, 3, 1], [0, 1, 2, 1, 3, 1],
                [0, 1, 0, 2, 3, 1], [0, 1, 2, 3, 4, 1], [0, 1, 0, 1, 0, 2],
                [0, 1, 2, 1, 0, 2], [0, 1, 0, 2, 0, 2], [0, 1, 2, 3, 0, 2], 
                [0, 1, 2, 0, 1, 2], [0, 1, 0, 2, 1, 2], [0, 1, 2, 3, 1, 2], 
                [0, 1, 2, 0, 3, 2], [0, 1, 2, 1, 3, 2], [0, 1, 0, 2, 3, 2],
                [0, 1, 2, 3, 4, 2], [0, 1, 2, 1, 0, 3], [0, 1, 0, 2, 0, 3], 
                [0, 1, 2, 3, 0, 3], [0, 1, 2, 0, 1, 3], [0, 1, 0, 2, 1, 3], 
                [0, 1, 2, 3, 1, 3], [0, 1, 2, 0, 2, 3], [0, 1, 0, 1, 2, 3], 
                [0, 1, 2, 1, 2, 3], [0, 1, 2, 3, 2, 3], [0, 1, 2, 3, 4, 3], 
                [0, 1, 2, 3, 0, 4], [0, 1, 2, 3, 1, 4], [0, 1, 2, 3, 2, 4], 
                [0, 1, 2, 0, 3, 4], [0, 1, 2, 1, 3, 4], [0, 1, 0, 2, 3, 4], 
                [0, 1, 2, 3, 4, 5]]