import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.decomposition import PCA
from snf import compute
from sklearn import cluster
from scipy.spatial import distance
from sklearn.utils.validation import check_array
# from construct_simulation_data import construct_simData
from munkres import Munkres
import time
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, KFold

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test


# read multi-omics data, transpose so that rows are samples and columns are features
def load_omics_data(omics_files):
    num_omics = len(omics_files)
    omics_list = []
    for i in range(num_omics):
        temp_omics = pd.read_csv(omics_files[i], header=0, index_col=0).T
        omics_list.append(temp_omics)
    return omics_list


# normalization
def normalize_matrix(omics_list, type='min-max'):
    num_omics = len(omics_list)
    retained_omics_list = []
    for i in range(num_omics):
        temp_omics = omics_list[i]
        if type == 'z-score':  # z-score normalization
            new_omics = preprocessing.scale(temp_omics, axis=0)
            retained_omics_list.append(new_omics)
        elif type == 'min-max':  # min-max normalization
            new_omics = preprocessing.minmax_scale(temp_omics, axis=0)
            retained_omics_list.append(new_omics)
        else:
            print("Error! required z-score or min-max")

    return retained_omics_list

# KNN constraint matrix for single omics (built with Euclidean distance)
def knn_kernel(data, k=10):
    # set sample weight to 1 for each row
    num_samples = data.shape[0]
    Dis_Mat = distance.cdist(data, data)
    kernel = np.ones_like(Dis_Mat)
    sort_dist = np.sort(Dis_Mat, axis=1)
    threshold = sort_dist[:, k].reshape(-1, 1)
    sig = (Dis_Mat <= np.repeat(threshold, num_samples, axis=1))
    # kernel = sig * kernel
    kernel = sig * kernel - np.identity(num_samples)
    return kernel

# fused KNN constraint matrix across multi-omics
def get_fused_kernel(omics_list, neighbor_num='default'):
    kernel_list = []
    num_samples = omics_list[0].shape[0]
    fused_kernel = np.zeros((num_samples, num_samples))
    if neighbor_num == 'default':
        neighbor_num = round(num_samples / 10)
        if neighbor_num < 25:
            neighbor_num = 25
        elif neighbor_num > 50:
            neighbor_num = 50
    for i in range(len(omics_list)):
        omics_kernel = knn_kernel(omics_list[i], k=neighbor_num)
        fused_kernel += omics_kernel
        kernel_list.append(omics_kernel)

    fused_kernel = np.ones((num_samples, num_samples)) * (fused_kernel > 0)
    return fused_kernel


# build vanilla auto-encoder
class AE_build(nn.Module):
    def __init__(self, AE_dims):
        super(AE_build, self).__init__()

        assert isinstance(AE_dims, list)
        self.encoder = nn.Sequential(
            nn.Linear(AE_dims[0], AE_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(AE_dims[1], AE_dims[2]),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(AE_dims[2], AE_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(AE_dims[1], AE_dims[0]),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

# auto-encoder with self-expression layer
class AE_Self_build(nn.Module):
    def __init__(self, AE_Self, num_samples, fused_kernel):
        super(AE_Self_build, self).__init__()
        self.AE_Self = AE_Self
        self.coef = nn.Parameter(1.0e-4*torch.ones((num_samples, num_samples), dtype=torch.float32))
        self.fused_kernel = fused_kernel


    def forward(self,x):
        coef=self.coef   #-torch.diag(torch.diag(self.coef))
        coef=coef * torch.tensor(self.fused_kernel>0,dtype=torch.float32)
        z = self.AE_Self.encoder(x)
        z_self=torch.matmul(coef, z)
        x_recon = self.AE_Self.decoder(z_self)
        return z, z_self, x_recon, coef

# self-supervised auto-encoder with self-expression
class Supervise_AE_self_build(nn.Module):
    def __init__(self, AE, num_samples, fused_kernel):
        super(AE_Self_build, self).__init__()
        self.AE=AE
        self.coef=nn.Parameter(1.0e-4*torch.ones((num_samples, num_samples), dtype=torch.float32))
        self.fused_kernel = fused_kernel

    def forward(self,x):
        coef=self.coef-torch.diag(torch.diag(self.coef))
        coef=coef * torch.tensor(self.fused_kernel>0,dtype=torch.float32)
        # fused_kernel = knn_kernel(x,k=26)
        # coef = coef * torch.tensor(fused_kernel>0,dtype=torch.float32)
        z = self.AE.encoder(x)
        z_self=torch.matmul(coef, z)
        x_recon = self.AE.decoder(z_self)
        return z, z_self,x_recon, coef

# train vanilla auto-encoder
def train_AE(model, X, epochs, lr, device, show_freq=-1):
    # ensure input is Tensor
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(np.array(X), dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_record = []
    for epoch in range(epochs):
        Z, X_recon = model.forward(X)
        loss = F.mse_loss(X_recon, X, reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_record.append(loss)
        if (epoch % show_freq == 0 or epoch == epochs - 1) & (show_freq != -1):
            # reconstruction loss
            print('Epoch {}: Loss_XX: {}'.format(epoch, loss))
            print(X.numpy()[100, 10:14], X_recon.detach().numpy()[100, 10:14])

    return loss_record



# fine-tune AE with self-expression layer
def train_AE_Self(model,  X, epochs, lr=1e-3, weight_xx=1.0, weight_self=1.0,
                  weight_coef=1.0, device='cpu'):
    # ensure input is Tensor
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(np.array(X), dtype=torch.float32, device=device)

    Total_Loss=[]
    Loss_XX=[]
    Loss_ZZ=[]
    Loss_coef=[]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        Z,Z_self,X_recon,coef=model.forward(X)

        loss_XX=F.mse_loss(X_recon, X, reduction='mean')
        loss_ZZ=F.mse_loss(Z_self, Z, reduction='mean')
        loss_coef = torch.sum(torch.pow(coef,2))
        total_loss = weight_xx * loss_XX \
                     + weight_self * loss_ZZ\
                     + weight_coef * loss_coef
        Total_Loss.append(total_loss)
        Loss_XX.append(loss_XX)
        Loss_ZZ.append(loss_ZZ)
        Loss_coef.append(loss_coef)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch==epochs-1:
            z_coef=coef

    return Total_Loss,Loss_XX,Loss_ZZ,Loss_coef,z_coef


# train self-supervised AE with self-expression
def train_Sup_AE_Self(model,  X, epochs, loss_sup,lr=1e-3, weight_xx=1.0, weight_self=1.0,
                  weight_coef=1.0, weight_sup=1.0,device='cpu'):
    # ensure input is Tensor
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(np.array(X), dtype=torch.float32, device=device)


    Total_Loss=[]
    Loss_XX=[]
    Loss_ZZ=[]
    Loss_sup=[]
    Loss_coef=[]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        Z,Z_self,X_recon,coef=model.forward(X)

        loss_XX=F.mse_loss(X_recon, X, reduction='mean')
        loss_ZZ=F.mse_loss(Z_self, Z, reduction='mean')
        loss_coef = torch.sum(torch.pow(coef,2))
        total_loss_sup = weight_xx * loss_XX \
                     + weight_self * loss_ZZ\
                     + weight_coef * loss_coef\
                     # + weight_sup * loss_sup
        Total_Loss.append(total_loss_sup)
        Loss_XX.append(loss_XX)
        Loss_sup.append(loss_sup)
        Loss_ZZ.append(loss_ZZ)
        Loss_coef.append(loss_coef)

        optimizer.zero_grad()
        total_loss_sup.backward()
        optimizer.step()

        if epoch==epochs-1:
            z_coef=coef


    return Total_Loss,Loss_XX, Loss_ZZ,Loss_sup,Loss_coef,z_coef




# build ensemble self-supervised sparse constraint matrix
def  get_labels_fused_ensm(y_labels,matrix_con):
    num_samples=len(y_labels[0])
    matrix_all = np.zeros((num_samples, num_samples))
    # set distance to 0 for samples with different labels
    for m in range(len(y_labels)):
        matrix_one = np.ones((num_samples, num_samples))
        for i in range(len(y_labels[m])):
            for j in range(len(y_labels[m])):
                if y_labels[m][i] != y_labels[m][j]:
                    matrix_one[i, j] = 0
        matrix_all=matrix_all+matrix_one

    matrix_all_0=matrix_all * (matrix_all>matrix_con)
    np.fill_diagonal(matrix_all_0, 0)

    return matrix_all_0

# train self-expression AE for each omics
def AE_self_fun(num_omics, AE_dims, omics_list, epochs, lr, num_samples, weight_xx, weight_self, weight_coef, device, show_freq):
    # ================================= Autoencoder pre-training ====================================
    for k in range(num_omics):
        print("the {}th omics: {}".format(k, omics_names[k]))  # the kth omics data
        pre_ae_model = pretrained_model_dir + '/{}-omics ae.pkl'.format(omics_names[k])
        if os.path.exists(pre_ae_model):
            continue
        print(" Pretraining  Autoencoder...")
        AE = AE_build(AE_dims=AE_dims[k])
        AE.to(device)
        loss_record = train_AE(AE, omics_list[k], epochs, lr=lr,
                               device=device, show_freq=show_freq)
        print('Pretrained AE loss:',loss_record)
        torch.save(AE.state_dict(), pre_ae_model)

    # ================================= SelfExpression Autoencoder ====================================
    epochs = 101
    Z_omics = []
    Z_coef = []
    fused_kernel = get_fused_kernel(omics_list)
    for k in range(num_omics):
        print("the {}th omics: {}".format(k, omics_names[k]))  # the kth omics data
        self_model = self_model_dir + '/{}-omics ae.pkl'.format(omics_names[k])
        AE_Self = AE_build(AE_dims=AE_dims[k])
        ae_state_dict = torch.load('Pretrained_AE/{}-omics ae.pkl'.format(omics_names[k]))
        AE_Self.load_state_dict(ae_state_dict)

        AE_Selfexpression = AE_Self_build(AE_Self, num_samples, fused_kernel)

        Total_Loss, Loss_XX, Loss_ZZ, Loss_coef, z_coef = train_AE_Self(AE_Selfexpression, omics_list[k], epochs, lr=lr,
                                                               weight_xx=weight_xx,
                                                               weight_self=weight_self, weight_coef=weight_coef,
                                                               device='cpu')

        Z_coef.append(z_coef.cpu().detach().numpy())  # reconstruction coefficient for each omics

        if not isinstance(omics_list[k], torch.Tensor):
            X = torch.tensor(np.array(omics_list[k]), dtype=torch.float32, device=device)
        z,_,_,_ = AE_Selfexpression.forward(X)
        z = z.cpu().detach().numpy()
        Z_omics.append(z)

        print('Total loss:', Total_Loss)
        print('AE loss:', Loss_XX)
        print('Self-reconstruction loss:', Loss_ZZ)
        print('Self-expression loss:', Loss_coef)
        torch.save(AE_Self.state_dict(), self_model)

    return Z_coef,Z_omics

# train fused AE module
def AE_fusion_fun(AE_dims_fusion,X_omics,epochs,lr,num_samples,weight_xx,weight_self, weight_coef,device):
    # ================================= Autoencoder pre-training ====================================

    print("the fusion_omics omics: train")  # the kth omics data
    fusion_ae_model = fusion_model_dir + '/Fusion_AE-omics ae.pkl'
    if os.path.exists(fusion_ae_model):
        print('Pretrained model already exists')
    else:
        Fusion_AE = AE_build(AE_dims=AE_dims_fusion)
        Fusion_AE.to(device)
        loss_record = train_AE(Fusion_AE, X_omics, epochs, lr=lr,
                               device=device, show_freq=-1)
        print('Pretrained fusion AE loss:', loss_record)
        torch.save(Fusion_AE.state_dict(), fusion_ae_model)

    # ================================= SelfExpression Autoencoder ====================================
    epochs = 101
    fusion_self_model = fusion_self_model_dir + '/Fusion_Self_AE ae.pkl'
    fused_kernel = get_fused_kernel(omics_list)
    print("the fusion omics: train")
    AE_Self = AE_build(AE_dims=AE_dims_fusion)
    ae_state_dict = torch.load(fusion_ae_model)
    AE_Self.load_state_dict(ae_state_dict)

    Fusion_AE_Self = AE_Self_build(AE_Self, num_samples, fused_kernel)

    Total_Loss, Loss_XX, Loss_ZZ, Loss_coef, z_coef = train_AE_Self(Fusion_AE_Self, X_omics, epochs, lr=lr,
                                                                    weight_xx=weight_xx, weight_self=weight_self,
                                                                    weight_coef=weight_coef, device='cpu')

    Z_coef=z_coef.cpu().detach().numpy()  # reconstruction coefficient for fused omics

    if not isinstance(X_omics, torch.Tensor):
        X = torch.tensor(np.array(X_omics), dtype=torch.float32, device=device)
    z,_,_,_ = Fusion_AE_Self.forward(X)
    Z_omics = z.cpu().detach().numpy()


    print('Total loss:', Total_Loss)
    print('AE loss:', Loss_XX)
    print('Self-reconstruction loss:', Loss_ZZ)
    print('Self-expression loss:', Loss_coef)
    torch.save(Fusion_AE_Self.state_dict(), fusion_self_model)

    return Z_coef, Z_omics

# ensemble classifier training and label prediction
def self_supervised_fun(X_train, X_test, y_train, y_test):
    # KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    y_pred_knn = knn_classifier.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print("KNN accuracy:", accuracy_knn)
    y_pred_knn_all = knn_classifier.predict(X)

    # Naive Bayes classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    y_pred_nb = nb_classifier.predict(X_test)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print("Naive Bayes accuracy:", accuracy_nb)
    y_pred_nb_all = nb_classifier.predict(X)

    # Decision Tree classifier
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    y_pred_dt = dt_classifier.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print("Decision Tree accuracy:", accuracy_dt)
    y_pred_dt_all = dt_classifier.predict(X)

    # SVM classifier
    svm_classifier = svm.SVC(kernel='linear', C=4)
    svm_classifier.fit(X_train, y_train)
    y_pred_svm = svm_classifier.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print("SVM accuracy:", accuracy_svm)
    y_pred_svm_all = svm_classifier.predict(X)

    y_labels = [labels_pred, y_pred_knn_all, y_pred_nb_all, y_pred_dt_all, y_pred_svm_all]
    loss_sup = ((1 - accuracy_svm) + (1 - accuracy_dt) + (1 - accuracy_nb) + (1 - accuracy_knn)) / 4.0

    return y_labels,loss_sup

# train self-supervised AE with ensemble sparse constraint
def AE_Sup_fun(fused_kernel,X_omics,loss_sup,lr,weight_xx,weight_self, weight_coef,weight_sup, cluster_num,device):

    epochs = 101

    sup_ae_model = fusion_self_model_dir + '/Fusion_Self_AE ae.pkl'
    AE_sup = AE_build(AE_dims=AE_dims_fusion)

    AE_Sup = AE_Self_build(AE_sup, num_samples, fused_kernel)
    sup_state_dict = torch.load(sup_ae_model)
    AE_Sup.load_state_dict(sup_state_dict)

    Total_Loss, Loss_XX, Loss_ZZ, Loss_sup, Loss_coef, z_coef = train_Sup_AE_Self(AE_Sup, X_omics, epochs, loss_sup,lr=lr,
                                                                   weight_xx=weight_xx,
                                                                   weight_self=weight_self, weight_coef=weight_coef,
                                                                   weight_sup=weight_sup,
                                                                   device='cpu')

    Z_coef=z_coef.cpu().detach().numpy()  # fused reconstruction coefficient

    if not isinstance(X_omics, torch.Tensor):
        X = torch.tensor(np.array(X_omics), dtype=torch.float32, device=device)
    z, _, _, _ = AE_Sup.forward(X)
    Z_omics = z.cpu().detach().numpy()


    print('Total loss:', Total_Loss)
    print('AE loss:', Loss_XX)
    print('Self-reconstruction loss:', Loss_ZZ)
    print('Prediction loss:', Loss_sup)
    print('Self-expression loss:', Loss_coef)

    Z_coef[Z_coef < 0] = 0
    Z_coef_all_C = 0.5 * (Z_coef + Z_coef.T)

    labels_pred = cluster.spectral_clustering(Z_coef_all_C, n_clusters=cluster_num)


    return  labels_pred, Z_omics

if __name__ == "__main__":

    start = time.time()
    pretrained_model_dir = 'Pretrained_AE'
    if not os.path.exists(pretrained_model_dir):
        os.makedirs(pretrained_model_dir)
    self_model_dir= 'SelfExpression_AE'
    if not os.path.exists(self_model_dir):
        os.makedirs(self_model_dir)
    fusion_model_dir = 'Fusion_AE'
    if not os.path.exists(fusion_model_dir):
        os.makedirs(fusion_model_dir)
    fusion_self_model_dir = 'Fusion_Self_AE'
    if not os.path.exists(fusion_self_model_dir):
        os.makedirs(fusion_self_model_dir)

    # ================================= load data ====================================
    omics_names = ['Methy', 'Mirna', 'Gene']
    omics_files = [r"D:\ML\DLSFIS\多组学数据\BIC\BREAST_Methy_Expression.csv"
        , r"D:\ML\DLSFIS\多组学数据\BIC\BREAST_Mirna_Expression.csv"
        ,r"D:\ML\DLSFIS\多组学数据\BIC\BREAST_Gene_Expression.csv"]
    omics_list = load_omics_data(omics_files=omics_files)  # load data

    # number of features
    for i in range(3):
        print('Number of features:', omics_list[i].shape[1])

    print("Number of samples:", omics_list[0].shape[0])

    # ================================= preprocessing data ====================================
    omics_list = normalize_matrix(omics_list, type='min-max')  # normalize
    # ================================= parameter setting ====================================
    num_omics = len(omics_list)
    num_samples = omics_list[0].shape[0]
    AE_dims = [[2000, 1000, 500], [885, 600, 500], [2000, 1000, 500]]  # AE nodes: Methy; miRNA; Gene
    AE_dims_fusion = [1500, 1000, 500]   # AE nodes for fusion module
    matrix_con= 3       # ensemble self-supervised constraint threshold
    epochs = 2000
    lr = 1e-5
    weight_xx = 1.0     # AE reconstruction error weight
    weight_self = 0.2   # self-expression reconstruction error weight under KNN constraint
    weight_coef = 1.0   # self-expression coefficient norm weight
    weight_sup = 0.2   # self-expression reconstruction error weight under ensemble self-supervision
    device = 'cpu'
    show_freq = -1
    df_clusternum=pd.DataFrame()
    clusternum_list=[]
    pval_list=[]
    for cluster_num in range(2,7):
        # self-expression training
        Z_coef, Z_omics = AE_self_fun(num_omics, AE_dims, omics_list, epochs, lr, num_samples, weight_xx, weight_self,
                                      weight_coef, device, show_freq)

        # multi-omics fusion
        X_omics = np.hstack((Z_omics[0], Z_omics[1], Z_omics[2]))
        Z_coef_fusion, Z_omics_fusion = AE_fusion_fun(AE_dims_fusion, X_omics, epochs, lr, num_samples, weight_xx,
                                                      weight_self, weight_coef, device)

        # spectral clustering
        Z_coef_fusion[Z_coef_fusion < 0] = 0
        Z_coef_fusion_C = 0.5 * (Z_coef_fusion + Z_coef_fusion.T)
        labels_pred = cluster.spectral_clustering(Z_coef_fusion_C, n_clusters=cluster_num)

        # compute survival p-value
        data_sur = pd.read_csv('D:\ML\DLSFIS\多组学数据\BIC\BREAST_Survival.csv', header=0, index_col=0)
        data_sur['Label'] = labels_pred
        results_0 = multivariate_logrank_test(data_sur['Survival'], data_sur['Label'], data_sur['Death'])

        # ensemble classification
        X = Z_omics_fusion
        y = labels_pred
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        y_labels, loss_sup = self_supervised_fun(X_train, X_test, y_train, y_test)


        # self-supervised training
        fused_kernel = get_fused_kernel(omics_list)  # fused KNN constraint matrix across multi-omics

        best_score = silhouette_score(Z_omics_fusion, labels_pred)
        best_labels = labels_pred.copy()
        temp_labels_pred = labels_pred.copy()
        for i in range(30):
            # optimize self-expression layer with self-supervision
            y_labels_kernel = get_labels_fused_ensm(y_labels, matrix_con)  # ensemble self-supervised sparse constraint matrix
            fused_kernel_sup = y_labels_kernel + fused_kernel   # combine ensemble self-supervised and fused KNN constraint
            labels_pred, Z_omics = AE_Sup_fun(fused_kernel_sup, X_omics, loss_sup, lr, weight_xx, weight_self,
                                              weight_coef, weight_sup, cluster_num, device)

            #Performing internal validation via the silhouette coefficient
            current_score = silhouette_score(Z_omics_fusion, labels_pred)
            
            if current_score > best_score:
                best_score = current_score
                best_labels = labels_pred.copy()
                temp_labels_pred = labels_pred.copy()
            else:
                labels_pred = temp_labels_pred  
                break

            # re-run ensemble classification
            X = Z_omics
            y = labels_pred
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            y_labels, loss_sup = self_supervised_fun(X_train, X_test, y_train, y_test)


        # final survival p-value
        data_sur_final = pd.read_csv(fr'D:\ML\DLSFIS\多组学数据\BIC\BREAST_Survival.csv', header=0, index_col=0)
        data_sur_final['Label'] = labels_pred
        #data_sur_final['Label'] = pd.Series(labels_pred, index=data_sur_final.index)
        results_final = multivariate_logrank_test(data_sur_final['Survival'],
                                                  data_sur_final['Label'],
                                                  data_sur_final['Death'])
        log10P = math.log10(results_final.p_value)
        print(f'Final cluster={cluster_num}, P-value={results_final.p_value}, log10P={log10P}')
        clusternum_list.append(cluster_num)
        pval_list.append(results_final.p_value)
    df_clusternum['cluster_num'] = clusternum_list
    df_clusternum['p_val'] = pval_list
    df_clusternum.to_csv('D:\Paper  Code\experiment.csv', index=True)
