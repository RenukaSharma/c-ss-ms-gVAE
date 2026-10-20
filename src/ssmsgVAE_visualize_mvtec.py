import click
from cv2 import NORMAL_CLONE
import torch
import logging
import random
import numpy as np
from networks.main import build_autoencoder
import sys
from PIL import Image
import os
import cv2
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid, plot_matrix_grid
from DeepSAD import DeepSAD
from datasets.main import load_dataset
import torchvision.transforms as transforms
from sklearn.manifold import TSNE

# model_path = "/home1/renuka/AD/results_UNet_bayesian_MVTec_alphaNbeta/MVTec-carpet_w_classifier/BS-128/n_known_outlier_classes_1/ratio_l_0.49/recon_param_1/gamma_1/classifier_coeff_1/eta_1/val_1e-1/baseline_A/run_0/model.tar" # full sup ss-gVAE # optimal_threshold = 329.87

model_path = "/home1/renuka/AD/results_UNet_bayesian_MVTec_alphaNbeta/MVTec-carpet_w_classifier/BS-128/n_known_outlier_classes_1/ratio_l_0/recon_param_1/gamma_1/classifier_coeff_1/eta_1/val_1e-1/baseline_A/run_0/model.tar" # 0 sup ss-gVAE # optimal_threshold = 329.87

eta = 1.0
recon_param = 1.0
device = "cuda:0"
# device = "cpu"

deepSAD = DeepSAD(eta, recon_param, xp_path="")
# net_name = "cifar10_LeNet"
net_name = "mvtec_net"
deepSAD.set_network(net_name)

deepSAD.load_model_stochastic(model_path=model_path, load_ae=True, map_location=device)

deepSAD.ae_net = deepSAD.ae_net.to(device)
deepSAD.ae_net.eval()

dataset_name = "mvtec"
category =  "carpet"
data_path = "/home1/renuka/AD/mvtec_AD_dataset/"+category+"-with-novel-ano-in-test"
normal_class = 0
known_outlier_class = 1
n_known_outlier_classes = 1
ratio_known_normal = 0
ratio_known_outlier = 0.49
ratio_pollution = 0
seed = 1 

image_folder = "/home1/renuka/AD/mvtec_anomaly_detection_textureANDobject/textures/carpet/test/color"
gt_folder = "/home1/renuka/AD/mvtec_anomaly_detection_textureANDobject/textures/carpet/ground_truth/color"
file_name = "000.png"

image_image = Image.open(os.path.join(image_folder, file_name)).convert('RGB') #.resize((32,32))
image = np.asarray(image_image)
print("Opened the image", file_name)

# reading the mask
mask_name = '000_mask.png' 
mask = np.asarray(Image.open(os.path.join(gt_folder, mask_name)).convert('L'))
print("Extracted the mask", mask_name)

mat = np.zeros((image.shape[0], image.shape[1]))
print("Size of matrix:", mat.shape)
count = 0
loopStartTime = time.time()

transform = transforms.Compose([      
            transforms.ToPILImage(),      
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
# running the model on a patch
input = transform(image)
# print("Shape of input",input.shape)

input = torch.unsqueeze(input, 0)
# print("Shape of input",input.shape)
input = input.to(device = 'cuda:0')
eps = 1e-5
reconImg, sampleRecon, samplingParasAll, out_class = deepSAD.ae_net(input, ablation_type='A', eps=eps)
# outputs_recons_mu, outputs_recons_beta = samplingParasAll['final']['mu'], samplingParasAll['final']['sigma']
outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta = samplingParasAll['final']['mu'], samplingParasAll['final']['sigma'], samplingParasAll['final']['beta']

dist_first = 0.0
dist_second = 0.0
for key in samplingParasAll['hidden'].keys():       
    para = samplingParasAll['hidden'][key]
    sample = para['sample']
    outputs_encoded_mu = para['mu']
    outputs_encoded_alpha = para['sigma']
    outputs_encoded_beta = para['beta']

print(outputs_encoded_mu.shape)
outputs_encoded_mu = np.squeeze(outputs_encoded_mu.cpu().detach().numpy())

train_labels = np.array([1])
scores = torch.norm(sample, dim=1) ** 2
scores = scores.detach().numpy()

predicted_labels = np.zeros_like(train_labels)
for i in range(scores.shape[0]):
    if(np.log(scores[i]) > np.log(scores).mean()):
        predicted_labels[i]=1
    else:
        predicted_labels[i]=0



# scores = torch.log(torch.norm(sample, dim=1) ** 2)
# scores = (torch.norm(sample, dim=1) ** 2)
scores = outputs_recons_mu ** 2
scores = torch.squeeze(scores)

mat = scores.cpu().detach().numpy()
mat = mat.mean(axis=0)
print("mat.shape",mat.shape)        

# if(scores >= 234.858):
#     mat[x_start + int(patch_size/2.0), y_start + int(patch_size/2.0)] = 1
# else:
#     mat[x_start + int(patch_size/2.0), y_start + int(patch_size/2.0)] = 0

# mat[x_start + int(patch_size/2.0), y_start + int(patch_size/2.0)] = scores


print("Matrix:\n",mat)
np.save("/home1/renuka/AD/results_bayesian_Corrosion-patchwise/heatmap_plots/"+file_index+"_mat", mat)
print("max val:", mat.max())
print("min val:", mat.min())
print("mean val:", mat.mean())
mat = (255.0 *(mat - mat.min())/(mat.max() - mat.min())).astype('uint8')

# plt.imshow(mat, interpolation='nearest')

# imgplot = plt.imshow(mat, interpolation="nearest")
# plt.axis('off')
plt.imsave("/home1/renuka/AD/results_bayesian_Corrosion-patchwise/heatmap_plots/"+file_index+"_heatmap.png", mat)
# plt.colorbar()
# default colormap for plt is virdis
plt.close()

# sys.exit()
