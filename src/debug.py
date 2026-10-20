from networks.cifar10_LeNet_vae_skip_conn_everywhere2 import CIFAR10_LeNet_Autoencoder
import torch 

if __name__ == '__main__':

    ae = CIFAR10_LeNet_Autoencoder()
    x = torch.randn(2,3, 32, 32)
    ae(x)