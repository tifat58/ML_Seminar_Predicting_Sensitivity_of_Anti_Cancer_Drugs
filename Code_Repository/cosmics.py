from __future__ import print_function
#import tensorflow as tf
import pandas as pd
import numpy as np
import collections
#import matplotlib.pyplot as plt
#from tkinter import *
#from sklearn.model_selection import KFold
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import f_classif



#read feature file, transpose and discard first column
#read response file
#feature selection with anova

#feat = pd.read_excel('../data/cnv_tim.xlsx')
#feat = pd.read_excel('../data/mutation_final.xlsx')
#feat = feat
#feat = feat[1:]

def matching():
    #get column x
    resp = pd.read_excel('../data/response_T_final.xlsx')
    resp = resp.T
    one = resp.iloc[:,1]
    one.to_frame()

    #drop all NaN

    one.dropna(how='any', inplace=True)

    resp = one
    #match cosmic ids
    indexf = list(feat.index)
    indexr = list(resp.index)
    print(indexf)
    print(indexr)

    for i in range(len(indexf)):
        if (indexf[i] not in indexr):
            feat.drop(index=indexf[i], inplace=True)
        if ('.1' in str(indexf[i])) and (int(str(indexf[i][:-2])) in list(feat.index)):
            s = str(indexf[i][:-2])
            feat.drop(index=int(s), inplace=True)

    indexf = list(feat.index)

    for j in range(len(indexr)):
        if (indexr.count(indexr[j]) > 1) or (indexr[j] not in indexf):
            resp.drop(index=indexr[j], inplace=True)
        if ('.1' in str(indexr[j])) and (int(str(indexr[j][:-2])) in list(resp.index)):
            s = str(indexr[j][:-2])
            resp.drop(index=int(s), inplace=True)

    Y = resp.as_matrix()

    X = feat

def sel():
    print(len(Y))

    x = feat.sum(axis=0)
    # print(sorted(x))

    ind = list(feat.columns)
    print(len(ind))

    for i in range(1, len(ind)):
        if (x[i] < 5):
            feat.drop(columns=ind[i], inplace=True, axis=1)

    #print(feat.sum(axis=0))
    print(X)
    print(X.columns)
    data_kv = []
    for i in range(len(Y)):
        print(list(X.iloc[i]))
        print(Y[i])
        val = X.iloc[i]
        z = Y[i]
        np.append(X.iloc[i], z)
        data_kv.append(X.iloc[i])
    data_kv = np.array(data_kv)
    print('data kv ', data_kv)



def cnv():
    print("features", len(X.columns))

    for i in range(1, len(X.columns)):
       # print(X[X.columns[i]])

        x = feat.sum(axis=0)
        print(sorted(x))
        ind = list(feat.columns)

        for i in range(len(ind)):
            if x[i] < 200:
                feat.drop(columns=ind[i], inplace=True, axis=1)
        print("Features selected: ", len(feat.columns))
        no = len(feat.columns)
        return feat, no



def read_cnv(feat, genelist):
    genes = set()
    cosmics = set()
    cos = collections.defaultdict(list)
    for row in feat.values:
        if (row[1] in genelist):
            #print(row[1])
            genes.add(str(row[1]))
            cosmics.add(row[2])

    print("no genes ", len(genes))
    print("no samples ", len(cosmics))
    arr = []
    for i in range(len(genes)):
        arr.append([0] * (len(cosmics)))

    #print(arr)

    genes = list(genes)
    cosmics = list(cosmics)

    for row in feat.values:
        if (row[1] in genelist):
            if(row[3] == 1):
                #print(row[1])
                arr[genes.index(str(row[1]))][cosmics.index(row[2])]= 1

    df = pd.DataFrame(arr, columns=cosmics, index=genes )
    writer = pd.ExcelWriter('cnv_tim_final.xlsx')
    df.to_excel(writer)
    writer.save()



def cnv_selection():
    genelist = []
    with open("../data/TCGA_Pathway_CNV_Amp.csv") as infile:
        for line in infile:
            if line:
                entry = line.split(",")
                gene = entry[0]
                genelist.append(gene)

    with open("../data/TCGA_Pathway_CNV_Dels.csv") as infile:
        for line in infile:
            if line:
                entry = line.split(",")
                gene = entry[0]
                genelist.append(gene)

    return genelist


#genelist = cnv_selection()
#read_cnv(feat, genelist)


def mut_selection():
    genelist = []
    infile = pd.read_excel("../data/mut_drivers.xlsx")
    for entry in infile.values:
        genelist.append(entry)
    print(len(genelist))
    return genelist



#read_cnv(feat, genelist)

def sel_mut(genelist, conv):
    print(len(feat.columns))
    arr = []
    genes = []

    for val in feat.index:
        #print(val)
        if conv[str(val)] in genelist:
            #print(list(feat[val]))
            #feat.drop(columns=val, inplace=True, axis=1)
            arr.append(list(feat.loc[val]))
            genes.append(conv[val])
        #print("Features selected: ", len(feat.columns))

    print(arr)
    print("cols", genes)
    print(feat.index)
    df = pd.DataFrame(arr, columns=feat.columns, index=genes)
    print(df.columns)
    #df = df.T
    print('neu', df.columns)
    print(len(arr))
    writer = pd.ExcelWriter('mutation_final_selected.xlsx')
    df.to_excel(writer)
    writer.save()


def gene_ens():
    conv = {}
    infile = pd.read_excel("../data/symbol_ensemble.xlsx")
    for row in infile.values:
        conv[str(row[1])] = str(row[0])
    #print(conv)
    return conv
'''
#conv = gene_ens()
#genelist = mut_selection()
#sel_mut(genelist, conv)


import sys
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# read feature file, transpose and discard first column
# read response file
# feature selection with anova


# 1 argument: features, not important for mutation data
# 2 argument: which drug
# 3 argument: 'mut' if features are mutation data

nfeat = int(sys.argv[1])
# print(nfeat)

# get drug
resp = pd.read_excel('../data/response_T_final.xlsx')
# print(resp)
resp = resp.T
name = list(resp)[int(sys.argv[2])]
print('Drugname: ', list(resp)[int(sys.argv[2])])
one = resp.iloc[:, int(sys.argv[2])]
one.to_frame()

# read feature data
feat = pd.read_excel('../data/expression_final4.xlsx')
mut = pd.read_excel('../data/mutation_final_selected.xlsx')
# feat = pd.read_excel('../data/methylation.xlsx')
cnv = pd.read_excel('../data/cnv_tim_final.xlsx')
# feat3 = feat3.T[1:]
feat = feat.T
mut = mut.T
cnv = cnv.T
print('finished reading')

# drop all NaN
one.dropna(how='any', inplace=True)
resp = one

# match cosmic ids, drop the ones that do not occur in both
indexf = list(feat.index)
#print(indexf)
indexr = list(resp.index)

for i in range(len(indexf)):
    if (indexf[i] not in indexr):
        feat.drop(index=indexf[i], inplace=True)
    if ('.1' in str(indexf[i])) and (int(str(indexf[i][:-2])) in list(feat.index)):
        s = str(indexf[i][:-2])
        feat.drop(index=int(s), inplace=True)

indexf = list(feat.index)

for j in range(len(indexr)):
    if (indexr.count(indexr[j]) > 1) or (indexr[j] not in indexf):
        resp.drop(index=indexr[j], inplace=True)
    if ('.1' in str(indexr[j])) and (int(str(indexr[j][:-2])) in list(resp.index)):
        s = str(indexr[j][:-2])
        resp.drop(index=int(s), inplace=True)

Y = resp.as_matrix()

if len(sys.argv) == 3:
    selector = SelectKBest(f_classif, k=nfeat)
    selector.fit_transform(feat, resp)
    idx_sel = selector.get_support(indices=True)

    feat_new = feat[idx_sel]
    #pd.DataFrame(feat, )#, index=ind, columns=cols)


feat_result = pd.concat([feat, mut, cnv], join='inner', axis=1)
feat_result = feat_result.dropna(how='any')


X = feat_result
nfeat = len(X.columns)
print("after dropout: " + str(nfeat))

print(X)

data_kv = []
if len(sys.argv) > 2:
    for i in range(len(Y)):
        data_kv.append(np.append(X.iloc[i], Y[i]))
#    data_kv.append(np.append(X[X.columns[i]], Y[i]))
else:
    for i in range(len(Y)):
        data_kv.append(np.append(X[i], Y[i]))
data_kv = np.array(data_kv)

def shuffle_train_data(X_train, Y_train):
    """called after each epoch"""
    #perm = np.random.permutation(len(Y_train))
    #Xtr_shuf = X_train[perm]
    #Ytr_shuf = Y_train[perm]

    return np.random.permutation(X_train)#.as_matrix())

#test = np.array([1,2,3,4,5])
#print(shuffle_train_data(test, resp))
'''

print(np.logsp)
