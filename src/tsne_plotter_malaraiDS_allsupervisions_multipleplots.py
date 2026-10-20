import tkinter
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
from matplotlib import cm
import numpy as np
import click
import torch
import logging
import random
import numpy as np
import PIL
from PIL import Image

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid, plot_matrix_grid
from DeepSAD import DeepSAD
from datasets.main import load_dataset
import torchvision.transforms as transforms

from datetime import datetime

timestamp_ = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

matplotlib.rcParams['legend.fontsize'] = 15
matplotlib.rcParams['legend.handletextpad']=0.01

dataset_name = 'malaria_dataset'
data_path = "../../nanofibre/"
# net_name = "nanofibre_vae"
net_name = "cifar10_LeNet"
xp_path = "/home/renuka/AD/mvtec_dataset"
normal_class = 0
n_known_outlier_classes = 1
known_outlier_class = 1
ratio_known_normal = 0
ratio_known_outlier = 0.2
ratio_pollution = 0
seed = 0


load_model = "/home/renuka/AD/results_bayesian_Malaria-patchwise/BS-128/no-noise_0.1-pollution_inv-loss-noveltest_w_classifier_corrected_Jun28/n_known_outlier_classes_1/ratio_l_0.2/recon_param_1/latent_param_1/classifier_coeff_1.0/eta_1/val_1e-1/baseline_A/run_1/model.tar" # ours with classifier at 0.2 supervision

eta = 1.0
recon_param = 1.0
# device = "cuda:7"
device = "cpu"

deepSAD = DeepSAD(eta, recon_param, xp_path=xp_path)
deepSAD.set_network(net_name)

deepSAD.load_model(model_path=load_model, load_ae=True, map_location=device)

deepSAD.ae_net = deepSAD.ae_net.to(device)
deepSAD.ae_net.eval()

dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                        ratio_known_normal, ratio_known_outlier, ratio_pollution,
                        random_state=np.random.RandomState(seed), length=5000)

train_labels = dataset.test_set.targets
print(type(train_labels))

final_train_data = torch.stack([I for I in dataset.test_set.tensor_batch])
final_train_data = final_train_data
train_labels = train_labels
print(final_train_data.shape)
final_train_data = final_train_data.to(device)
"""
final_train_data = torch.stack([I for I in dataset.train_set.tensor_batch])
final_train_data = final_train_data
train_labels = train_labels
print(final_train_data.shape)
final_train_data = final_train_data.to(device)
"""

# outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample, class_out = deepSAD.ae_net(final_train_data)

outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample, class_out = deepSAD.ae_net(final_train_data)


print(outputs_encoded_mu.shape)

# np.save("/home/renuka/syntheticExperiment_EXPonly_AnoDetect_patch-based_with-GGnoise_10kand2p5k/features/tsne_test_feats_E.npy", outputs_encoded_mu.cpu().detach().numpy())
# np.save("/home/renuka/syntheticExperiment_EXPonly_AnoDetect_patch-based_with-GGnoise_10kand2p5k/features/test_labels_E.npy", np.array(train_labels))

# outputs_encoded_mu = np.load("/home/renuka/syntheticExperiment_EXPonly_AnoDetect_patch-based_with-GGnoise_10kand2p5k/features/tsne_test_feats_A.npy")
# train_labels = np.load("/home/renuka/syntheticExperiment_EXPonly_AnoDetect_patch-based_with-GGnoise_10kand2p5k/features/test_labels_A.npy")

"""
np.save("/home/renuka/syntheticExperiment_EXPonly_AnoDetect_patch-based_with-GGnoise_10kand2p5k/features/tsne_train_feats_A.npy", outputs_encoded_mu.cpu().detach().numpy())
np.save("/home/renuka/syntheticExperiment_EXPonly_AnoDetect_patch-based_with-GGnoise_10kand2p5k/features/train_labels_A.npy", np.array(train_labels))

outputs_encoded_mu = np.load("/home/renuka/syntheticExperiment_EXPonly_AnoDetect_patch-based_with-GGnoise_10kand2p5k/features/tsne_train_feats_A.npy")
train_labels = np.load("/home/renuka/syntheticExperiment_EXPonly_AnoDetect_patch-based_with-GGnoise_10kand2p5k/features/train_labels_A.npy")
"""
train_labels = np.array(train_labels)

sample_size = 2500
################### sampling ####################
# s = np.arange(train_labels.shape[0])
# np.random.shuffle(s) # shuffling step
# sample_size = 1000
# train_labels = train_labels[s[0:sample_size]]
# sample = sample[s[0:sample_size]]
# outputs_encoded_mu = outputs_encoded_mu[s[0:sample_size]]
#################################################

tsne = TSNE(2, verbose=1)
# tsne = TSNE(3, verbose=1)
# tsne_proj = tsne.fit_transform(outputs_encoded_mu) #### enable it
# tsne_proj = tsne.fit_transform(outputs_encoded_mu.cpu().detach().numpy())
# print(tsne_proj)
# Plot those points as a scatter plot and label them based on the pred labels
# cmap = cm.get_cmap('tab20')
############## Generating the scores ##############
scores = torch.norm(sample, dim=1) ** 2
scores = scores.detach().numpy()

predicted_labels = np.zeros_like(train_labels)
for i in range(scores.shape[0]):
    if(np.log(scores[i]) > np.log(scores).mean()):
        predicted_labels[i]=1
    else:
        predicted_labels[i]=0

# predicted_labels = np.zeros_like(train_labels)
# for i in range(scores.shape[0]):
#     if(np.log(scores[i]) > np.log(scores).mean()):
#         predicted_labels[i]=1
#     else:
#         predicted_labels[i]=0
# for i in range(scores.shape[0]):
#     if(scores[i] > 178.0):
#         predicted_labels[i]=1
#     else:
#         predicted_labels[i]=0

####################################################

######################
# fig, ax = plt.subplots(figsize=(8,8))
fig, ax = plt.subplots(2,2)
# tsne_proj = tsne.fit_transform(outputs_encoded_mu.cpu().detach().numpy()) 
# np.save("/home/renuka/AD/malaria_dataset/tsne_viz/tsne_proj_E_sup0.2_"+timestamp_+".npy", tsne_proj)
tsne_proj = np.load("/home/renuka/AD/malaria_dataset/tsne_viz/tsne_proj_A_sup0.2_2021_09_23-08:32:55_PM.npy")
num_categories = 2

list_1 = []
list_2 = []
list_3 = []
list_4 = [] # for legend

for i in range(predicted_labels.shape[0]):
    list_1.append(tsne_proj[i,0]) # x's
    list_2.append(tsne_proj[i,1]) # y's
    
    if (predicted_labels[i]==train_labels[i]): 
        legend_="True" 
        marker_dict_1 ={0:"<", 1:">"}  
        list_3.append(marker_dict_1[train_labels[i]])
    else:
        legend_="False"
        marker_dict_2 ={0:"P", 1:"X"}
        list_3.append(marker_dict_2[train_labels[i]])
     
    if predicted_labels[i]==0:
        label_ = "Inliers"
    else:
        label_ = "Outliers"
    # print("Appending",legend_,label_)
    list_4.append(legend_+" "+label_)

flag_tp=0
count_tp = 0
flag_tn=0
count_tn = 0
flag_fp=0
count_fp=0
flag_fn=0
count_fn=0
# colors=["#0000FF", "#00FF00", "#FF0066"]
color_scheme = {"tp":"olive", "tn":"salmon", "fp":"indigo", "fn":"deepskyblue"}

marker_size = 30 # 100 was good


for x, y, m, l in zip(list_1, list_2, list_3, list_4):
    if l=="True Inliers":
        if count_tp < 500:
            ax[0,0].scatter([x], [y] , marker = m, label = l if flag_tp==0 else "", c = color_scheme["tp"], s= marker_size)
            # ax[1,1].scatter([x], [y] , marker = m, label = l if flag_tp==0 else "", c = color_scheme["tp"], s= marker_size)
            count_tp+=1
        flag_tp=1
    if l=="True Outliers":
        if count_tn < 500:
            ax[0,0].scatter([x], [y] , marker = m, label = l if flag_tn==0 else "", c = color_scheme["tn"], s= marker_size)
            count_tn+=1
        flag_tn=1
    

for x, y, m, l in zip(list_1, list_2, list_3, list_4):    
    if l=="False Inliers":        
        if(count_fp< 20):
            ax[0,0].scatter([x], [y] , marker = m, label = l if flag_fp==0 else "", c = color_scheme["fp"], s= marker_size)
            count_fp+=1
        flag_fp=1
        
    if l=="False Outliers":        
        if(count_fn< 20):
            ax[0,0].scatter([x], [y] , marker = m, label = l if flag_fn==0 else "", c = color_scheme["fn"], s= marker_size)        
            count_fn+=1
        flag_fn=1

####################################################
#### ax[0,1]

load_model = "/home/renuka/AD/results_bayesian_Malaria-patchwise/BS-128/no-noise_with0.1pollution_inv-loss-noveltest_May22/n_known_outlier_classes_1/ratio_l_0.2/recon_param_1/latent_param_0.5/eta_1/val_0.1/baseline_A/run_1/model.tar" # ours withOUT classifier at 0.2 supervision

eta = 1.0
recon_param = 1.0
# device = "cuda:7"
device = "cpu"
net_name = "cifar10_LeNet_wo_classifier"
deepSAD = DeepSAD(eta, recon_param, xp_path=xp_path)
deepSAD.set_network(net_name)

deepSAD.load_model(model_path=load_model, load_ae=True, map_location=device)

deepSAD.ae_net = deepSAD.ae_net.to(device)
deepSAD.ae_net.eval()

outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample = deepSAD.ae_net(final_train_data)
print(outputs_encoded_mu.shape)
train_labels = np.array(train_labels)

sample_size = 2500
################### sampling ####################
# s = np.arange(train_labels.shape[0])
# np.random.shuffle(s) # shuffling step
# sample_size = 1000
# train_labels = train_labels[s[0:sample_size]]
# sample = sample[s[0:sample_size]]
# outputs_encoded_mu = outputs_encoded_mu[s[0:sample_size]]
#################################################

tsne = TSNE(2, verbose=1)
############## Generating the scores ##############
scores = torch.norm(sample, dim=1) ** 2
scores = scores.detach().numpy()

predicted_labels = np.zeros_like(train_labels)
for i in range(scores.shape[0]):
    if(np.log(scores[i]) > np.log(scores).mean()):
        predicted_labels[i]=1
    else:
        predicted_labels[i]=0


# tsne_proj = tsne.fit_transform(outputs_encoded_mu.cpu().detach().numpy()) 
# np.save("/home/renuka/AD/malaria_dataset/tsne_viz/tsne_proj_Ablation_sup0.2_"+timestamp_+".npy", tsne_proj)
tsne_proj = np.load("/home/renuka/AD/malaria_dataset/tsne_viz/tsne_proj_Ablation_sup0.2_2021_09_24-03:13:58_AM.npy")
num_categories = 2

list_1 = []
list_2 = []
list_3 = []
list_4 = [] # for legend

for i in range(predicted_labels.shape[0]):
    list_1.append(tsne_proj[i,0]) # x's
    list_2.append(tsne_proj[i,1]) # y's
    
    if (predicted_labels[i]==train_labels[i]): 
        legend_="True" 
        marker_dict_1 ={0:"<", 1:">"}  
        list_3.append(marker_dict_1[train_labels[i]])
    else:
        legend_="False"
        marker_dict_2 ={0:"P", 1:"X"}
        list_3.append(marker_dict_2[train_labels[i]])
     
    if predicted_labels[i]==0:
        label_ = "Inliers"
    else:
        label_ = "Outliers"
    # print("Appending",legend_,label_)
    list_4.append(legend_+" "+label_)

flag_tp=0
count_tp = 0
flag_tn=0
count_tn = 0
flag_fp=0
count_fp=0
flag_fn=0
count_fn=0
# colors=["#0000FF", "#00FF00", "#FF0066"]
color_scheme = {"tp":"olive", "tn":"salmon", "fp":"indigo", "fn":"deepskyblue"}

marker_size = 30 

for x, y, m, l in zip(list_1, list_2, list_3, list_4):
    if l=="True Inliers":
        if count_tp < 500:
            ax[0,1].scatter([x], [y] , marker = m, label = l if flag_tp==0 else "", c = color_scheme["tp"], s= marker_size)
            count_tp+=1
        flag_tp=1
    if l=="True Outliers":
        if count_tn < 500:
            ax[0,1].scatter([x], [y] , marker = m, label = l if flag_tn==0 else "", c = color_scheme["tn"], s= marker_size)
            count_tn+=1
        flag_tn=1
    

for x, y, m, l in zip(list_1, list_2, list_3, list_4):    
    if l=="False Inliers":        
        if(count_fp< 40):
            ax[0,1].scatter([x], [y] , marker = m, label = l if flag_fp==0 else "", c = color_scheme["fp"], s= marker_size)
            count_fp+=1
        flag_fp=1
        
    if l=="False Outliers":        
        if(count_fn< 40):
            ax[0,1].scatter([x], [y] , marker = m, label = l if flag_fn==0 else "", c = color_scheme["fn"], s= marker_size)        
            count_fn+=1
        flag_fn=1
####################################################
## ax[1,0] -DeepSAD
load_model = "/home1/renuka/AD/results_bayesian_Malaria-patchwise/BS-128/deepsad-with-noise-no-pollution/n_known_outlier_classes_1/ratio_l_0.2/eta_1/baseline_deepsad/run_1/model.tar"

eta = 1.0
recon_param = 1.0
# device = "cuda:7"
device = "cpu"
net_name = "cifar10_LeNet_fromdeepsad"
deepSAD = DeepSAD(eta, recon_param, xp_path=xp_path)
deepSAD.set_network(net_name)

deepSAD.load_model(model_path=load_model, load_ae=True, map_location=device)

deepSAD.ae_net = deepSAD.ae_net.to(device)
deepSAD.ae_net.eval()

outputs_encoded_mu, recons = deepSAD.ae_net(final_train_data)
print(outputs_encoded_mu.shape)
train_labels = np.array(train_labels)

sample_size = 2500
################### sampling ####################
# s = np.arange(train_labels.shape[0])
# np.random.shuffle(s) # shuffling step
# sample_size = 1000
# train_labels = train_labels[s[0:sample_size]]
# sample = sample[s[0:sample_size]]
# outputs_encoded_mu = outputs_encoded_mu[s[0:sample_size]]
#################################################

tsne = TSNE(2, verbose=1)
############## Generating the scores ##############
scores = torch.norm(sample, dim=1) ** 2
scores = scores.detach().numpy()

predicted_labels = np.zeros_like(train_labels)
for i in range(scores.shape[0]):
    if(np.log(scores[i]) > np.log(scores).mean()):
        predicted_labels[i]=1
    else:
        predicted_labels[i]=0


# tsne_proj = tsne.fit_transform(outputs_encoded_mu.cpu().detach().numpy()) 
# np.save("/home/renuka/AD/malaria_dataset/tsne_viz/tsne_proj_deepsad_sup0.2_"+timestamp_+".npy", tsne_proj)
tsne_proj = np.load("/home/renuka/AD/malaria_dataset/tsne_viz/tsne_proj_deepsad_sup0.2_2021_09_24-03:33:39_AM.npy")
num_categories = 2

list_1 = []
list_2 = []
list_3 = []
list_4 = [] # for legend

for i in range(predicted_labels.shape[0]):
    list_1.append(tsne_proj[i,0]) # x's
    list_2.append(tsne_proj[i,1]) # y's
    
    if (predicted_labels[i]==train_labels[i]): 
        legend_="True" 
        marker_dict_1 ={0:"<", 1:">"}  
        list_3.append(marker_dict_1[train_labels[i]])
    else:
        legend_="False"
        marker_dict_2 ={0:"P", 1:"X"}
        list_3.append(marker_dict_2[train_labels[i]])
     
    if predicted_labels[i]==0:
        label_ = "Inliers"
    else:
        label_ = "Outliers"
    # print("Appending",legend_,label_)
    list_4.append(legend_+" "+label_)

flag_tp=0
count_tp = 0
flag_tn=0
count_tn = 0
flag_fp=0
count_fp=0
flag_fn=0
count_fn=0
# colors=["#0000FF", "#00FF00", "#FF0066"]
color_scheme = {"tp":"olive", "tn":"salmon", "fp":"indigo", "fn":"deepskyblue"}

marker_size = 30 

for x, y, m, l in zip(list_1, list_2, list_3, list_4):
    if l=="True Inliers":
        if count_tp < 1000:
            ax[1,0].scatter([x], [y] , marker = m, label = l if flag_tp==0 else "", c = color_scheme["tp"], s= marker_size)
            count_tp+=1
        flag_tp=1
    if l=="True Outliers":
        if count_tn < 500:
            ax[1,0].scatter([x], [y] , marker = m, label = l if flag_tn==0 else "", c = color_scheme["tn"], s= marker_size)
            count_tn+=1
        flag_tn=1
    

for x, y, m, l in zip(list_1, list_2, list_3, list_4):    
    if l=="False Inliers":        
        if(count_fp< 60):
            ax[1,0].scatter([x], [y] , marker = m, label = l if flag_fp==0 else "", c = color_scheme["fp"], s= marker_size)
            count_fp+=1
        flag_fp=1
        
    if l=="False Outliers":        
        if(count_fn< 60):
            ax[1,0].scatter([x], [y] , marker = m, label = l if flag_fn==0 else "", c = color_scheme["fn"], s= marker_size)        
            count_fn+=1
        flag_fn=1
####################################################


# ax.scatter(tsne_proj[indices,0], tsne_proj[indices,1], label = label_, alpha=0.5, marker = marker_dict_1[lab])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())  
# predicted X, actual Y; Inliers- Predicted Inliers, Outliers- Predicted Outliers, True- 

##### To keep legends on ##################
# lgnd = ax.legend([by_label['True Inliers'],by_label['False Outliers'],by_label['True Outliers'],by_label['False Inliers']],['Predicted Inlier, Actual Inlier','Predicted Outlier, Actual Inlier','Predicted Outlier, Actual Outlier','Predicted Inlier, Actual Outlier'], markerscale=2, loc='lower right') # legend, remove if you don't want the legend

# lgnd = ax[0,0].legend([by_label['True Inliers'],by_label['False Outliers'],by_label['True Outliers'],by_label['False Inliers']],['Predicted Inlier, Actual Inlier','Predicted Outlier, Actual Inlier','Predicted Outlier, Actual Outlier','Predicted Inlier, Actual Outlier'], markerscale=2) # legend, remove if you don't want the legend
# lgnd.legendHandles[0]._sizes = [60]
# lgnd.legendHandles[1]._sizes = [60]
# lgnd.legendHandles[2]._sizes = [60]
# lgnd.legendHandles[3]._sizes = [60]
###########################################

# Hide axes ticks
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
# ax.legend(fontsize='large', markerscale=2)
# plt.show()

ax[0,1].set_xticks([])
ax[0,1].set_yticks([])

ax[1,0].set_xticks([])
ax[1,0].set_yticks([])

ax[0, 0].set_title("RU-VAE")
ax[0, 1].set_title("RU-VAE-Ablated")
ax[1, 0].set_title("DeepSAD")
# ax[1, 1].set_title("Legends")
######################

########################
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# num_categories = 2
# for lab in range(num_categories):
#     indices = train_labels==lab
#     ax.scatter(tsne_proj[indices,0], tsne_proj[indices,1],tsne_proj[indices,2], label = lab ,alpha=0.5)
# ax.legend(fontsize='large', markerscale=2)

# # import pickle
# # pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))

# plt.show()
# # Hide axes ticks
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
########################

plt.grid(False)
plt.axis('off')
plt.tight_layout()
# Hide grid lines
ax[0,0].grid(False)

legends_only = Image.open('/home/renuka/EXPonly_AnoDetect_patch-based_with-GGnoise_10kand2p5k_withClassifier_rough_bestForMalAndBUI_classifier/src/just_legends_.png')
legends_only = legends_only.resize((800,300))
height = legends_only.size[1]
width = legends_only.size[0]
fig.figimage(legends_only, fig.bbox.xmax +width//2.15, fig.bbox.ymax -height//1.5)
fig.savefig("/home/renuka/AD/malaria_dataset/tsne_viz/tsne_plots_RU-VAE.png", dpi=300, bbox_inches='tight')
# plt.savefig("/home/renuka/syntheticExperiment_EXPonly_AnoDetect_patch-based_with-GGnoise_10kand2p5k/plotting_visualization/tsne_test_withmarkers_E_sup0.49_"+timestamp_+".png")
# plt.savefig("/home/renuka/AD/malaria_dataset/plotsGenerated_May25_onwards/tsne_test_3d_1.fig")