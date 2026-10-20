
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class SkipBlock(nn.Module):

    def __init__(self, tag, inChannels, outChannels, skip=False, stochastic=False):
        super().__init__()
        
        self.tag = tag
        self.skip = skip
        self.stochastic = stochastic
        self.eps = 1e-5
        
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
            nn.LeakyReLU(inplace=True)
            )
            
            self.estimateSigma = nn.Sequential(
            nn.Conv2d(outChannels, outChannels, 1),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, 1),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(inplace=True)
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
        sigma = torch.exp(self.estimateSigma(fMap) + self.eps)
        sample = mu + sigma*torch.normal(torch.zeros_like(mu), torch.ones_like(sigma))
        return sample, mu, sigma
    
    
    def downSample(self, x):
        # import ipdb; ipdb.set_trace()

        fMap = self.downsampleModule(x)            
        if self.stochastic:
            sample, mu, sigma = self.sample(fMap)
            # samplingParas = {'mu':mu, 'sigma':sigma}
            return fMap, mu, sigma, sample
        else:
            return fMap, None, None, None
        
    
    def upSample(self, fMap, skipInput):
        if self.skip:
            fMap = torch.cat([fMap, skipInput], dim=1)            
        upFMap = self.upSampleModule(fMap)
        return upFMap


class FinalLayer(nn.Module):
    def __init__(self, stochastic=False):
        self.stochastic = stochastic
        self.eps = 1e-5
        nrChannels = 3
        if self.stochastic:
            self.estimateMu = nn.Sequential(
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
            
            )
        
    
    def sample(self, fMap):
        mu = self.estimateMu(fMap)
        sigma = torch.exp(self.estimateSigma(fMap) + self.eps)
        sample = mu + sigma*torch.normal(torch.zeros_like(mu), torch.ones_like(sigma))
        return mu, sigma, sample
    
    def forward(self, fMap):
        if self.stochastic:
            mu, sigma, sample = self.sample(fMap)
            # samplingParas = {'mu':mu, 'sigma':sigma}
            return mu, sigma, sample
        else:
            return fMap, None, None


class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        inChannels = config['INCHANNELS']
        outChannels = config['OUTCHANNELS']
        skips = config['SKIPS'] 
        stochStatus = config['STOCHASTIC']

        self.nrBlocks = len(inChannels)
        self.skipBlocks = {}
        
        
        for i in range(self.nrBlocks):
            tag = self.getTag(i)
            self.skipBlocks[tag] = SkipBlock(tag=tag, inChannels = inChannels[i], outChannels = outChannels[i], skip = skips[i], stochastic=stochStatus[i])       
        
        self.finalLayer = nn.Sequential(
            nn.Conv2d(inChannels[0], inChannels[0], 1),
            nn.Sigmoid()
        )
        self.rep_dim = 128

        self.bn1d1 = torch.nn.BatchNorm1d(128)
        self.classifier_layer1 = nn.Linear(self.rep_dim, 16, bias=False)
        self.bn1d2 = torch.nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1, bias=False)        
        self.classifier_layer2 = nn.Linear(16, 1, bias=False)
    
    def getTag(self, idx):
        return '{:02d}'.format(idx)
    
    def forward(self, img, ablation_type='A', eps=1e-5):
        fMap = img
        fMapSkipAll = {}
        hiddenSamplingParasAll = {}
        # samplingParasAll = {}
        
        for i in range(self.nrBlocks):
            tag = self.getTag(i)
            fMap, mu, sigma, sample = self.skipBlocks[tag].downSample(fMap)
            # fMapSkipAll[tag] = fMapSkip
            # if samplingParas is not None:
            #     hiddenSamplingParasAll[tag] = samplingParas   
        
        encoding = fMap
       
        for i in range(self.nrBlocks -1,-1,-1):
            tag = self.getTag(i)
            fMap = self.skipBlocks[tag].upSample(fMap, fMapSkipAll[tag])
        
        recon_mu, recon_sigma, recon_sample = self.finalLayer(fMap)
        # samplingParasAll['hidden'] = hiddenSamplingParasAll
        # samplingParasAll['final'] = samplingParasFinal

        """
        Classifier results
        """
        class_out = self.bn1d1(sample)
        class_out = self.classifier_layer1(class_out)
        class_out = self.bn1d2(class_out)
        class_out = self.classifier_layer2(class_out)
        class_out = torch.sigmoid(class_out)
        
        """
        if self.training:        
            return reconImg,sampleRecon, samplingParasAll
        else:
            return encoding    
        """
        # return reconImg,sampleRecon, samplingParasAll  
        # return mu.cuda(), sigma.cuda(), recon_mu.cuda(), recon_sigma.cuda(), sample.cuda(), class_out.cuda()
        return mu, sigma, recon_mu, recon_sigma, sample, class_out
    
        
def getUNet(configFile, modelPath=None):
    with open(configFile, 'r') as openfile:
        config = json.load(openfile)
    
    Unet = UNet(config)
    
    if modelPath is not None:
        data = torch.load(modelPath)
        Unet.load_state_dict(data['state_dict'], strict=False)
        
    return Unet
                
                
if __name__ == "__main__":
    img = torch.rand((2,3,32,32))
    Unet = getUNet(os.path.join(os.path.dirname(__file__), 'configs', 'UNetConfig.json'))
    mu, sigma, recon_mu, recon_sigma, sample, class_out = Unet(img)