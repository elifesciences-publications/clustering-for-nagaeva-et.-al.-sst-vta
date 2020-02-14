# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:00:26 2017

@author: zubarei1
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler#, StandardScaler
from sklearn.model_selection import KFold
from sklearn import mixture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import openpyxl
path = '/m/nbe/work/zubarei1/collabs/elinas_APs/data_elina/'
import os
os.chdir(path)

raw_header = ['cell_id', 'cluster_id', 'Resting MP',    'Input resistance',
              'SAG DP',    'Rheobase current', 'RMP std',    'tau start',
              'tau stop',    '1_spk:Spike count', '1_spk:Delay',
              '1_spk:ADP amp', '1_spk:ADP lat', '1_spk:ADP angle', '1_spk:ADP norm',
              '1_spk:AP amplitude',
              '1_spk:AP HW',    '1_spk:Thresh amp',    '1_spk:AHP amp',
              '1_spk:AP decay time',    'sat:Spike count',  'sat:Current Step',
              'sat:Delay', 'sat:Fmax Init', 'sat:Fmax SS', 'sat:Adaptation',
              'sat:AP amp',    'sat:AP HW',    'sat:Thresh amp',    'sat:AHP amp',
              'sat:AP decay time']

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
tableau20=np.array([ (230,159,0), (0,158,115), (86,180,233),(128,128,128)])/255.

np.random.seed(42)


def read_xls(fname):
    wb = openpyxl.load_workbook(fname)
    ws = wb['2Sweeps_data']
    header = []
    t = []
    for i, row in enumerate(ws.rows):
        header.append(row[0])
        if i == 0:
            types = [c.value for c in row[1:]]
        elif i == 1:
            names = [c.value for c in row[1:]]
        elif row[0].value!=None:
            r = [c.value for c in row[1:]]
            t.append(r)
    return types, names, np.array(t,dtype=np.float32), header

def make_pretty(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="both", which="both", bottom="off", top="off",
                            labelbottom="on", left="off", right="off", labelleft="on")
    return ax

def preprocess(X, names, scaler, logcols, outcols, badfc=[4, 6, 12, 11]):
    nso = np.shape(X)[0]
    names = np.array(names)
    #remove nans

    #Log-transform Gamma-distributed variables add offset and noise for numerical stability
    X[:,[9,10,11,12]] += 1.+ np.random.exponential(scale=1, size=((X.shape[0]),4))
    X[:,logcols] = np.log2(X[:,logcols])

    nanrow, nancol = np.where(np.isnan(X))
    X = np.delete(X,np.unique(nancol),1)
    names = np.delete(names,np.unique(nanrow))
    #scale
    #X[:,19] = 1./X[:,19]
    X = scaler.transform(X)
    #remove outliers
    outrows = np.unique(np.where(np.abs(X[:,outcols])>5)[0])
    X = np.delete(X,outrows,0)
    names = np.delete(names,np.unique(outrows),0)
    #remove unwanted features
    X = np.delete(X,badfc,1)
    nsn = np.shape(X)[0]

    print('Dropred: %i' % (nso-nsn))
    return X, names, np.unique(list(nancol)+badfc), list(np.unique(nanrow))+list(np.unique(outrows))#, np.unique(np.array(outrows))


typ, names, X, h = read_xls(''.join([path+'juvenile/','juv_fixed.xlsx']))
X = X.T
print('juv:', X.shape)

typ, namest, Xt, _ = read_xls(''.join([path+'adults/','adults_fixed.xlsx']))
Xt = Xt.T
print('adults:', Xt.shape)


"""Preprocess"""
logcols = np.array([5,6,7,8,9,10,11,12,17,18,20,21,22,28]) #features to log_transform
outcols = [22, 23] #features to check for outliers (frequency estiates)
iqr = (25, 75)
scaler = RobustScaler(quantile_range=iqr)

badfc = [4, 6, 12, 11] # drop unused features that were poorly extracted by

#feature extraction pipeline
scaler.fit(X)

X, names, dropped_cols, droped_rows = preprocess(X, names, scaler, logcols,
                                                 outcols, badfc=badfc)
Xt, namest, _, droped_rowst = preprocess(Xt, namest, scaler,
                                        logcols, outcols, badfc=badfc)

scaler2 =  MinMaxScaler()#
X = scaler2.fit_transform(X)
Xt = scaler2.transform(Xt)

# Copy for visualiztion
X0 = X
X0t = Xt
print('Dropping unused/redundant features: ',', '.join([raw_header[i+2] for i in dropped_cols]))
header = np.delete(raw_header, np.unique(dropped_cols))
max_pca = .95 #percentage of variance explained

pca = PCA(n_components=max_pca, whiten=True)
X = pca.fit_transform(X)
Xt = pca.transform(Xt)
#

"""Model selection for number of PCs and clusters"""


kf = KFold(n_splits=5)
cv_pca_bic = []
holdout_pca_bic = []
min_bic = np.infty

for j in np.arange(1, X.shape[-1] + 1):
    X_param_search = X[:, :j]
    pca_bic = []

    n_clusters_range = np.arange(1, 7)
    for n_clusters in n_clusters_range:
        #  Fit a Gaussian mixture with EM
        bic_val = 0
        gmm = mixture.GaussianMixture(n_components=n_clusters,
                                      random_state=42,
                                      init_params='random',
                                      covariance_type='full')
        for train, val in kf.split(X_param_search):
            gmm.fit(X_param_search[train, :])
            bic_val += gmm.bic(X_param_search[val, :])/j
        if bic_val/5. < min_bic:
            min_bic = bic_val/5.
            n_pca = j
            n_comp = n_clusters
print('Optimal combination: n_pca:{}, n_clusters:{}'.format(n_pca, n_comp))

#%% Fit the model using potimal parameters
X = X[:, :n_pca]
Xt = Xt[:, :n_pca]
lowest_bic = np.infty
lowest_bic_tt = np.infty
lowest_bic_ever = np.infty
bic_train = []
bic_test = []
n_components_range = range(1, 7)

for n_components in n_components_range:
    gmm = mixture.GaussianMixture(n_components=n_components,
                                 random_state=42, init_params='random',
                                 covariance_type='full')
    bic_test0 = []
    bic_train0 = []
    for train, test in kf.split(X):
        gmm.fit(X[train,:])
        bic_train0.append(gmm.bic(X)/X.shape[0])
        bic_test0.append(gmm.bic(X[test,:])/X[test,:].shape[0])
    bic_train.append(np.mean(np.array(bic_train0)))
    bic_test.append(np.mean(np.array(bic_test0)))
    if bic_train[-1] < lowest_bic:
        lowest_bic = bic_train[-1]
        best_gmm_tr = gmm
    if bic_test[-1] < lowest_bic_tt:
        lowest_bic_tt = bic_test[-1]
        if lowest_bic_tt< lowest_bic_ever:
            best_gmm = gmm
bic_train = np.array(bic_train)
bic_test = np.array(bic_test)
bars = []



#%%
"""MODEL SELECTION FIGURE"""
plt.figure()
xpos = np.array(n_components_range) + .2 * (0 - 2)
bars.append(plt.bar(xpos, bic_train,
            width=.2, color=tableau20[0]))
xpos = np.array(n_components_range) + .2 * (1 - 2)
bars.append(plt.bar(xpos, bic_test,
                    width=.2, color=tableau20[1],edgecolor='none'))
plt.xticks(n_components_range)
plt.title('Parameter search for 2 PCs: 5-fold CV')
xpos = bic_test.argmin() + 1 - 0.14
plt.text(xpos, bic_test.min() * 0.97 + .03 * bic_test.max(), '*', fontsize=14)
plt.ylim([bic_train.min() * .95, bic_test.max()*1.05])
plt.xlim(0,6.5)
plt.xlabel('Number of Clusters')
plt.ylabel('BIC')
plt.show()
plt.legend(['BIC CV', 'BIC Holdout'],loc='upper left', frameon=False)
#
#%%
"""SCATTER PLOT"""
#Culster #0 is always the biggest
plt.figure()
clusters = np.argsort(best_gmm.weights_)[::-1]
Ztrain = clusters[best_gmm.predict(X)]
Ztest = clusters[best_gmm.predict(Xt)]

means = best_gmm.means_[clusters,:]
covs = best_gmm.covariances_[clusters,...]
weights = best_gmm.weights_[clusters]

ax = make_pretty(plt.gca())
xrange = np.linspace(np.min(X[:,0])*1.1,np.max(X[:,0])*1.1,100)
yrange = np.linspace(np.min(X[:,1])*1.1,np.max(X[:,1])*1.1,100)
gridx, gridy = np.meshgrid(xrange,yrange)
grid = np.dstack((gridx,gridy))

nk = best_gmm.n_components

"""CHANGE HERE FOR BETTER DESIGN"""
plt.scatter(X[:,0],X[:,1],c=tableau20[Ztrain],s=25,edgecolor='none',alpha=0.75)
plt.scatter(Xt[:,0],Xt[:,1],c=tableau20[Ztest],s=35,edgecolor='k',alpha=0.95)
from scipy.stats import multivariate_normal
for jj,i in enumerate(clusters):
    if best_gmm_tr.covariance_type == 'diag':
        mc = multivariate_normal(means[i,:2],np.diag(covs[i,:2]))#precs_
    else:
        mc = multivariate_normal(means[i,:2],covs[i,:2,:2])#precs_
    Z = mc.pdf(grid)
    plt.contour(xrange,yrange, Z,2,colors=tableau20[i][None,:])#, norm=LogNorm(vmin=1.0, vmax=1000.0),
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()





#%%
"""PLOT PCA COMPONENT CONTRIBUTIONS"""

plt.figure()
coef = np.transpose(pca.components_)[:,:2]
pve = pca.explained_variance_ratio_
cols = ['PC-'+str(x) for x in range(1,3)]
pc_infos = pd.DataFrame(coef, columns=cols, index=header[2:])
fig = plt.gcf()
clst_indx = [3,3,3,3,3,3,0,0,2,1,1,1,1,3,2,3,3,3,0,3,2,0,0,2,3,3,3,3,3,0,3,3,3]
alphas = 0.35*np.ones(len(clst_indx))
alphas[[7,8,9,10,11,12,13,14,18,20,21,22,23,29]]+=.5
clst_indx = np.delete(clst_indx,badfc)
alphas = np.delete(alphas,badfc)
colorsf = tableau20[clst_indx]
ax = make_pretty(plt.gca())


for idx in range(len(pc_infos["PC-1"])):
	x = pc_infos["PC-1"][idx]
	y = pc_infos["PC-2"][idx]
	plt.plot([0.0,x],[0.0,y],color=colorsf[idx],linewidth=3,alpha=0.5)
	plt.plot(x, y,marker='o',color=colorsf[idx],markeredgecolor='w',alpha=alphas[idx])
	plt.annotate(pc_infos.index[idx], xy=(x,y),color=colorsf[idx],fontsize=14,alpha=alphas[idx])
plt.xlabel("PC-1 (%s%%)" % str(pve[0])[:4].lstrip("0."))
plt.ylabel("PC-2 (%s%%)" % str(pve[1])[:4].lstrip("0."))
plt.show()
#%%
"""PLOT RAW FEATURES"""

shapes = ['s','s','s'] #'*','d','o','x','^'

plt.figure(figsize=(12,9))

Xo = np.vstack([X0 - X0.mean(0)[None,...], X0t - X0t.mean(0)[None,...]])
labels = np.concatenate([Ztrain,Ztest])
clust_means = [Xo[labels==i,:].mean(0) for i in range(nk)]
clust_stds = [Xo[labels==i,:].std(0) for i in range(nk)]

ns = [np.sqrt(np.sum(labels==i)) for i in range(nk)]

ax = make_pretty(plt.gca())

#plt.hold(True)
"""CHANGE HERE FOR BETTER DESIGN"""
[plt.errorbar(np.arange(Xo.shape[-1]),clust_means[i],yerr=clust_stds[i]/ns[i]*2,
    color=tableau20[i], marker=shapes[i], linestyle='None', markersize=ns[i]*1.5,
    linewidth=2, alpha=0.75, markeredgecolor='w')
    for i in range(best_gmm.n_components)]
ax = plt.gca()
pos = ax.get_position()
pos.y0 +=.12
ax.set_position(pos)
plt.legend()
plt.xlim(-1,len(clust_means[0]+1))

plt.xticks(np.arange(Xo.shape[-1]),header,fontsize=14,rotation='vertical')

plt.ylabel('Arbitrary Units')

plt.show()


import csv
def save_labels(filename,names,Z,data,header):
    with open((filename), 'a') as csv_file:
        data = np.concatenate([names[:,None],Z[:,None],data],axis=1)
        print(data.shape)
        #head = ['name', 'label'] + [*header]
        writer = csv.DictWriter(csv_file,fieldnames=header)
        writer.writeheader()
        for d in data:
            log ={k:v for k,v in zip(header,d)}
            writer.writerow(log)

#save_labels('juvenile.csv',names,Ztrain,X0,header)
#save_labels('adults.csv',namest,Ztest,X0t,header)