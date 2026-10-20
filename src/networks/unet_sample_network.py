import torch.nn as nn
import torch
import torchvision

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=( 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        # shape = x.shape
        # H = shape[-2]
        # W = shape[-1]
        # enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        # return enc_ftrs
        return x


class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128), dec_chs=( 128, 64), num_class=1, retain_dim=False, out_sz=(32,32)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim

        self.rep_dim=128
        self.bn2d1 = torch.nn.BatchNorm2d(128)
        self.classifier_layer1 = nn.Linear(self.rep_dim, 16, bias=False)
        self.bn1d2 = torch.nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1, bias=False)        
        self.classifier_layer2 = nn.Linear(16, 1, bias=False)

    def forward(self, x, ablation_type='A', eps=1e-5):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, out_sz)
        # return out

        """
        Classifier results
        """
        # class_out = self.bn2d1(out)
        # class_out = self.classifier_layer1(out)
        # class_out = self.bn1d2(class_out)
        # class_out = self.classifier_layer2(class_out)
        # class_out = torch.sigmoid(class_out)
        
        # import ipdb; ipdb.set_trace()
        print("$$$$$$$$$$$$$$$$$$$$$$Returning from UNet")

        # return enc_ftrs[-1], torch.ones_like(enc_ftrs[-1]) * (2 ** 0.5), torch.ones_like(enc_ftrs[-1]) * 2.0, out, torch.ones_like(out) * (2 ** 0.5), torch.ones_like(out) *  2.0, out
        return enc_ftrs[-1],  torch.ones_like(enc_ftrs[-1]) * 2.0, out, torch.ones_like(out) *  2.0, out


if __name__ == '__main__':

    encoder = Encoder()
    # input image
    x    = torch.randn(1, 3, 32, 32)
    ftrs = encoder(x)
    for ftr in ftrs: print(ftr.shape)

    decoder = Decoder()
    x = torch.randn(1, 128, 10, 10)
    decoder(x, ftrs[::-1][1:]).shape
    print (x.shape)

    unet = UNet()
    x    = torch.randn(1, 3, 32, 32)
    print("unet(x).shape=",unet(x).shape)
