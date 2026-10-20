
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

def log_gg(x, mu,  beta, tau, eps):
    # temp = ((-1) * (torch.abs(mu - x))**(beta+tau)) - torch.log(torch.tensor(2.0)) - torch.log( torch.lgamma(1 + 1.0/(beta+tau)))
    
    temp = (
            - (((torch.abs(mu - x) ) ** 2 + eps) ** (0.5 * (tau + beta)))
            + (torch.log(tau + beta))
            - (torch.lgamma(1 / (tau + beta)))
            # - (torch.log( alpha))
            - (torch.log(torch.tensor(2.0)))
    )
    # print("torch.abs(x)=", torch.abs(x))
    # print("beta+tau=", beta+tau)
    # print("(2 * torch.lgamma(1 + 1.0/(beta+tau)))=", (2 * torch.lgamma(1 + 1.0/(beta+tau))))
    # print("The value for temp is", temp)
    return torch.sum(temp, dim=tuple(range(1, mu.dim())))

def sample_ggd(mu, beta):
    gamma = Gamma(1/beta, 1)
    y = gamma.rsample()
    p = torch.ones_like(mu)
    s = torch.bernoulli(p) - 0.5
    return mu + 2*s*(y**(1/beta))

class SkipBlock(nn.Module):

    def __init__(self, tag, inChannels, outChannels, skip=False, stochastic=False):
        super().__init__()
        
        self.tag = tag
        self.skip = skip
        self.stochastic = stochastic
        
        self.downsampleModule = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, 3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, 3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
            )

        
        if stochastic:
            self.estimateMu = nn.Sequential(
            nn.Conv2d(outChannels, outChannels, 1),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, 1),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, 1),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(inplace=True)
            )
            
            self.estimateSigma = nn.Sequential(
            nn.Conv2d(outChannels, outChannels, 1),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, 1),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, 1),
            nn.BatchNorm2d(outChannels),
            # 2.0 * nn.Sigmoid()
            nn.Sigmoid()
            )
            
       
            
        if self.skip:
            upInchannels = 2*outChannels
        else:
            upInchannels = outChannels
        
        upOutChannels = inChannels
            
            
        
        self.upSampleModule = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(upInchannels, upOutChannels, 3, padding=1),
            nn.BatchNorm2d(upOutChannels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(upOutChannels, upOutChannels, 3, padding=1),
            nn.BatchNorm2d(upOutChannels),
            nn.LeakyReLU(inplace=True)
            )
        
    
    def sample(self, fMap):
        mu = self.estimateMu(fMap)
        sigma = torch.exp(self.estimateSigma(fMap))
        #sample = mu + sigma*torch.normal(torch.zeros_like(mu), torch.ones_like(sigma)).cuda()
        sample = sample_ggd(mu, sigma)
        return sample, mu, sigma
    
    
    def downSample(self, x):
        fMap = self.downsampleModule(x)            
        if self.stochastic:
            sample, mu, sigma = self.sample(fMap)
            samplingParas = {'mu':mu, 'sigma':sigma}
            return fMap, sample, samplingParas
        else:
            return fMap, fMap, None
        
    
    def upSample(self, fMap, skipInput):
        if self.skip:
            fMap = torch.cat([fMap, skipInput], dim=1)            
        upFMap = self.upSampleModule(fMap)
        return upFMap


class FinalLayer(nn.Module):
    def __init__(self, stochastic=False):
        super().__init__()
        self.stochastic = stochastic
        nrChannels = 1 # 1 for mnist, 3 for cifar10, ...
        if self.stochastic:
            self.estimateMu = nn.Sequential(
            nn.Conv2d(nrChannels, nrChannels, 1),
            nn.BatchNorm2d(nrChannels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(nrChannels, nrChannels, 1),
            nn.BatchNorm2d(nrChannels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(nrChannels, nrChannels, 1),
            nn.BatchNorm2d(nrChannels),
            nn.LeakyReLU(inplace=True)
            )
            
            self.estimateSigma = nn.Sequential(
            nn.Conv2d(nrChannels, nrChannels, 1),
            nn.BatchNorm2d(nrChannels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(nrChannels, nrChannels, 1),
            nn.BatchNorm2d(nrChannels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(nrChannels, nrChannels, 1),
            nn.BatchNorm2d(nrChannels),
            # 2.0 * nn.Sigmoid(),
            nn.Sigmoid()
            )
        
    
    def sample(self, fMap):
        mu = self.estimateMu(fMap)
        sigma = torch.exp(self.estimateSigma(fMap))
        #sample = mu + sigma*torch.normal(torch.zeros_like(mu), torch.ones_like(sigma)).cuda()
        sample = sample_ggd(mu, sigma)
        return sample, mu, sigma
    
    def forward(self, fMap):
        if self.stochastic:
            sample, mu, sigma = self.sample(fMap)
            samplingParas = {'mu':mu, 'sigma':sigma}
            return fMap, sample, samplingParas
        else:
            return fMap, fMap, None
            
        
    

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rep_dim = 128
        self.bn1d1 = torch.nn.BatchNorm1d(128)
        self.classifier_layer1 = nn.Linear(self.rep_dim, 16, bias=False)
        self.bn1d2 = torch.nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1, bias=False)        
        self.classifier_layer2 = nn.Linear(16, 1, bias=False)

    def forward(self, sample):
        sample = sample.view(-1,self.rep_dim)
        """
        Classifier results
        """
        class_out = self.bn1d1(sample)
        class_out = self.classifier_layer1(class_out)
        class_out = self.bn1d2(class_out)
        class_out = self.classifier_layer2(class_out)
        class_out = torch.sigmoid(class_out)
        return class_out


class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        inChannels = config['INCHANNELS']
        outChannels = config['OUTCHANNELS']
        skips = config['SKIPS'] 
        stochStatus = config['STOCHASTIC']

        self.nrBlocks = len(inChannels)
        self.skipBlocks = torch.nn.ModuleDict()
        
        
        for i in range(self.nrBlocks):
            tag = self.getTag(i)
            self.skipBlocks[tag] = SkipBlock(tag=tag, inChannels = inChannels[i], outChannels = outChannels[i], skip = skips[i], stochastic=stochStatus[i])

        #self.skipBlockLists = torch.nn.Module       
        
        self.finalLayer = FinalLayer(stochastic=True)

        self.classifier = Classifier()
        
    
    def getTag(self, idx):
        return '{:02d}'.format(idx)
    
    def forward(self, img, ablation_type='A', eps=1e-5):
        fMap = img
        fMapSkipAll = {}
        hiddenSamplingParasAll = {}
        samplingParasAll = {}
        
        for i in range(self.nrBlocks):
            tag = self.getTag(i)
            fMap, fMapSkip, samplingParas = self.skipBlocks[tag].downSample(fMap)
            fMapSkipAll[tag] = fMapSkip
            if samplingParas is not None:
                samplingParas['sample'] = fMapSkip
                hiddenSamplingParasAll[tag] = samplingParas   
        
        encoding = fMap
       
        for i in range(self.nrBlocks -1,-1,-1):
            tag = self.getTag(i)
            fMap = self.skipBlocks[tag].upSample(fMap, fMapSkipAll[tag])
        
        reconImg, sampleRecon, samplingParasFinal = self.finalLayer(fMap)
        samplingParasAll['hidden'] = hiddenSamplingParasAll
        samplingParasAll['final'] = samplingParasFinal
        class_out = self.classifier(encoding)
        
        # if self.training:        
        #     return reconImg,sampleRecon, samplingParasAll, class_out
        # else:
        #     return encoding    
        return reconImg,sampleRecon, samplingParasAll, class_out
    
    
        
def getUNet(configFile, modelPath=None):
    with open(configFile, 'r') as openfile:
        config = json.load(openfile)
    
    Unet = UNet(config)
    
    if modelPath is not None:
        data = torch.load(modelPath)
        Unet.load_state_dict(data['state_dict'], strict=False)
        
    return Unet
                
                
if __name__ == "__main__":
    img = torch.rand((2,3,32,32)).cuda()
    Unet = getUNet(os.path.join(os.path.dirname(__file__), 'configs', 'UNetConfig.json')).cuda()
    reconImg, sampleRecon, samplingParasAll, class_out = Unet(img)

    outputs_recons_mu, outputs_recons_beta = samplingParasAll['final']['mu'], samplingParasAll['final']['sigma']

    dist_first = 0.0
    dist_second = 0.0
    for key in samplingParasAll['hidden'].keys():       
        para = samplingParasAll['hidden'][key]
        sample = para['sample']
        outputs_encoded_mu = para['mu']
        outputs_encoded_beta = para['sigma']
        
        dist_first += log_gg(sample, outputs_encoded_mu,
                        outputs_encoded_beta, 1e-1, 1e-1)

        dist_second += 0.5 * (torch.norm(sample, dim=1) ** 2)
    dist_recon = 0.2 * (-1) * log_gg(img, outputs_recons_mu, 
                                outputs_recons_beta,
                                1e-1, 1e-1)
    """
    dist_first = log_gg(sample, outputs_encoded_mu, torch.ones_like(outputs_encoded_alpha),
                        torch.exp(outputs_encoded_beta), self.delta, self.tau, self.eps)
    dist_second = 0.5 * (torch.norm(sample, dim=1) ** 2)
    dist_recon = (-1) * log_gg(inputs, outputs_recons_mu, torch.ones_like(outputs_recons_alpha),
                                torch.exp(outputs_recons_beta),
                                self.delta, self.tau, self.eps)
    """
    dist = dist_first + dist_second + dist_recon
    print("dist=",dist)