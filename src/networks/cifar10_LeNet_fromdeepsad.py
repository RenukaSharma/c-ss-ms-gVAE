import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class CIFAR10_LeNet_fromdeepsad(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        # self.pool = nn.MaxPool2d(2, 2)

        # self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        # self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        # self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        # self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        # self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        # self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        # self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

        self.conv1 = nn.Conv2d(3, 32, 4, bias=False, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv2 = nn.Conv2d(32, 64, 4, bias=False, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv3 = nn.Conv2d(64, 128, 4, bias=False, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.fc2 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.fc3 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('leaky_relu'))


    def forward(self, x):
        # x = x.view(-1, 3, 32, 32)
        # x = self.conv1(x)
        # x = self.pool(F.leaky_relu(self.bn2d1(x)))
        # x = self.conv2(x)
        # x = self.pool(F.leaky_relu(self.bn2d2(x)))
        # x = self.conv3(x)
        # x = self.pool(F.leaky_relu(self.bn2d3(x)))
        # x = x.view(int(x.size(0)), -1)
        # x = self.fc1(x)
        # return x

        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = F.leaky_relu(self.bn2d1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2d2(x))
        x = self.conv3(x)
        x = F.leaky_relu(self.bn2d3(x))
        x = x.view(int(x.size(0)), -1)

        x_mu = self.fc1(x)

        return x_mu  # x_encoded


class CIFAR10_LeNet_Decoder_fromdeepsad(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        # self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        # nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        # self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        # nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        # self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        # nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        # self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        # nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.d1 = nn.Linear(self.rep_dim, 128*2*4*4)
        nn.init.xavier_uniform_(self.d1.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(128*2, 128, 3, 1)
        nn.init.xavier_uniform_(self.d2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn6 = nn.BatchNorm2d(128, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(128, 64, 3, 1)
        nn.init.xavier_uniform_(self.d3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn7 = nn.BatchNorm2d(64, 1.e-3)

        self.up3_mu = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3_mu = nn.ReplicationPad2d(1)
        self.d4_mu = nn.Conv2d(64, 32, 3, 1)
        nn.init.xavier_uniform_(self.d4_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn8_mu = nn.BatchNorm2d(32, 1.e-3)

        self.d5_mu = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.d5_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
        # x = F.leaky_relu(x)
        # x = self.deconv1(x)
        # x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        # x = self.deconv2(x)
        # x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        # x = self.deconv3(x)
        # x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        # x = self.deconv4(x)
        # x = torch.sigmoid(x)
        # return x

        h1 = self.relu(self.d1(x))
        h1 = h1.view(-1,128*2 , 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4_mu = self.leakyrelu(self.bn8_mu(self.d4_mu(self.pd3_mu(self.up3_mu(h3)))))
     
        x_mu = self.sigmoid(self.d5_mu(h4_mu))
        
        return x_mu


class CIFAR10_LeNet_Autoencoder_fromdeepsad(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = CIFAR10_LeNet_fromdeepsad(rep_dim=rep_dim)
        self.decoder = CIFAR10_LeNet_Decoder_fromdeepsad(rep_dim=rep_dim)

    def forward(self, x):
        y = self.encoder(x)
        x = self.decoder(y)
        return y, x
