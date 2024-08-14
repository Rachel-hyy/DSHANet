import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1),
                                               stride=(stride, stride), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x

class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class Decoder_GASN(nn.Module):
    def __init__(self, inplanes, neck_name='fpn+ppm+fuse'):
        super(Decoder_GASN, self).__init__()
        self.conv1 = Conv3Relu(inplanes * 16, inplanes * 8) 
        self.conv2 = Conv3Relu(inplanes * 8, inplanes * 4)  
        self.conv3 = Conv3Relu(inplanes * 4, inplanes * 2)  
        self.conv4 = Conv3Relu(inplanes * 2, inplanes)
        
        self.dr1 = DR(inplanes*8, inplanes)
        self.dr2 = DR(inplanes*4, inplanes)
        self.dr3 = DR(inplanes*2, inplanes)
        self.dr4 = DR(inplanes, inplanes)
        
        self.last_conv = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(inplanes),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(inplanes),
                                       nn.ReLU(),
                                       )
        self._init_weight()
        self.FAM_feat  = [inplanes*3, inplanes*2, inplanes*2, inplanes*1]
        self.FAM = FAM_layers(self.FAM_feat, in_channels=inplanes*3, batch_norm=True, dilation=True)
        self.FAM_feat2  = [inplanes*8, inplanes*4, inplanes*2, inplanes*1]
        self.FAM2 = FAM_layers(self.FAM_feat, in_channels=inplanes*8, batch_norm=True, dilation=True)
        self.FAM_output_layer = nn.Conv2d(inplanes, inplanes, kernel_size=1)

    def forward(self, fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4):
        change1_h, change1_w = fa1.size(2), fa1.size(3)

        change1 = self.conv1(torch.cat([fa1, fb1], 1))
        change2 = self.conv2(torch.cat([fa2, fb2], 1))
        change3 = self.conv3(torch.cat([fa3, fb3], 1))
        change4 = self.conv4(torch.cat([fa4, fb4], 1))

        x1 = self.dr1(change1)
        x2 = self.dr2(change2)
        x3 = self.dr3(change3)
        x4 = self.dr4(change4)

        x2 = F.interpolate(x2, size=(change1_h, change1_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(change1_h, change1_w), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(change1_h, change1_w), mode='bilinear', align_corners=True)
        
        x = torch.cat((x2, x3, x4), dim=1)

        x = x1 + self.FAM(x)

        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoderGASN(fc, BatchNorm):
    return Decoder_GASN(fc, BatchNorm)

def FAM_layers(cfg, in_channels=3, batch_norm=True, dilation=False):
    if dilation:
        d_rate = 3
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)   