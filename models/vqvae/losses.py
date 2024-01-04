import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

############### Perceptual Loss ###############
class Vgg16(nn.Module):
    def __init__(self, pretrained = True):
        super(Vgg16, self).__init__()
        self.vggnet = models.vgg16(pretrained)
        del(self.vggnet.classifier) # Remove fully connected layer to save memory.
        features = list(self.vggnet.features)
        self.layers = nn.ModuleList(features).eval() 
        
    def forward(self, x, stop_layer=3):
        results = []
        count = 0
        for ii,model in enumerate(self.layers):
            x = model(x)
            if ii in [3,8,15,22,29]:
                results.append(x) #(64,256,256),(128,128,128),(256,64,64),(512,32,32),(512,16,16)
                count += 1
                if stop_layer > 0 and count == stop_layer:
                    break
        return results

def perceptual_loss(vggnet, x, x_gt, layer=3):
    f_x = vggnet(x, layer)
    f_x_gt = vggnet(x_gt, layer)
    loss = 0.
    for i in range(layer):
        loss = loss + torch.mean((f_x[i] - f_x_gt[i])**2)
    return loss / layer

############### Discriminator Loss ###############
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sn=True, bn=True):
        super(ResBlock, self).__init__()
        self.bn = bn
        self.sn = sn
        if sn:
            self.conv0 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0))
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        else:
            self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        h_conv0 = self.conv0(x)
        if self.bn:
            h_conv1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
            h_conv2 = self.bn2(self.conv2(h_conv1))
        else:
            h_conv1 = F.relu(self.conv1(x), inplace=True)
            h_conv2 = self.conv2(h_conv1)
        out = F.relu(h_conv0 + h_conv2, inplace=True)
        return out

class Discriminator(nn.Module):
    def __init__(self, ndf=128, input_channels=3):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        # (128,128,3) -> (64,64,64)
        self.res1 = ResBlock(input_channels, ndf, bn=False)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # (64,64,64) -> (32,32,128)
        self.res2 = ResBlock(ndf, ndf*2, bn=False)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        # (32,32,128) -> (16,16,256)
        #self.res3 = ResBlock(ndf*2, ndf*4, bn=False)
        self.res3 = ResBlock(ndf*2, ndf*4, bn=False)
        self.conv_out = nn.Conv2d(in_channels=ndf*4, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_res1 = self.res1(x)
        h_pool1 = self.pool1(h_res1)
        h_res2 = self.res2(h_pool1)
        h_pool2 = self.pool2(h_res2)
        h_res3 = self.res3(h_pool2)
        out = self.conv_out(h_res3)
        return torch.sigmoid(out), out
    
def patch_discriminator_loss(disc, x_fake, x_real):
    # D Loss
    d_real, _ = disc(x_real)
    d_real_loss = nn.BCELoss()(d_real, torch.ones_like(d_real))
    d_fake, _ = disc(x_fake.detach())
    d_fake_loss = nn.BCELoss()(d_fake, torch.zeros_like(d_fake))
    d_loss = d_real_loss + d_fake_loss
    # G Loss
    g_fake, _ = disc(x_fake)
    g_loss = nn.BCELoss()(g_fake, torch.ones_like(g_fake))
    return d_loss, g_loss

if __name__ == "__main__":
    vgg_model = Vgg16()
    vgg_model = vgg_model.cuda()
    print(vgg_model.layers)
