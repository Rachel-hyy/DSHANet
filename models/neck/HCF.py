import torch
import torch.nn as nn
import torch.nn.functional as F

from models.block.Base import Conv3Relu
from models.block.Drop import DropBlock
from models.block.Field import PPM, ASPP, SPP
from models.neck.HCM import HCM


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChangeGuideModule(nn.Module):
    def __init__(self, in_dim):
        super(ChangeGuideModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()
        m_batchsize1, C1, height1, width1 = guiding_map0.size()
     

        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)

        guiding_map = F.sigmoid(guiding_map0)

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out

class HCFNeck(nn.Module):
    def __init__(self, inplanes, neck_name='hcf+ppm+fuse'):
        super().__init__()
        self.stage1_Conv1 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv1 = Conv3Relu(inplanes * 4, inplanes * 2) 
        self.stage3_Conv1 = Conv3Relu(inplanes * 8, inplanes * 4) 
        self.stage4_Conv1 = Conv3Relu(inplanes * 16, inplanes * 8)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.d2 = Conv3Relu(inplanes * 2, inplanes)
        self.d3 = Conv3Relu(inplanes * 4, inplanes)
        self.d4 = Conv3Relu(inplanes * 8, inplanes)
        self.final_Conv = Conv3Relu(inplanes * 4, 1)

        if "+ppm+" in neck_name:
            print("PPM is aviablei")
            self.expand_field = PPM(inplanes * 8)
        elif "+aspp+" in neck_name:
            self.expand_field = ASPP(inplanes * 8)
        elif "+spp+" in neck_name:
            self.expand_field = SPP(inplanes * 8)
        else:
            self.expand_field = None
        if "drop" in neck_name:
            rate, size, step = (0.15, 7, 30)
            self.drop = DropBlock(rate=rate, size=size, step=step)
        else:
            self.drop = DropBlock(rate=0, size=0, step=0)

        self.decoder = nn.Sequential(BasicConv2d(inplanes*8, inplanes, 3, 1, 1), nn.Conv2d(inplanes, 1, 3, 1, 1))

        self.decoder_final = nn.Sequential(BasicConv2d(inplanes*2, inplanes, 3, 1, 1), nn.Conv2d(inplanes, 1, 1))

        self.cgm_2 = ChangeGuideModule(inplanes*4)
        self.cgm_3 = ChangeGuideModule(inplanes*8)
        self.cgm_4 = ChangeGuideModule(inplanes*8)

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_module4 = BasicConv2d(inplanes*12, inplanes*8, 3, 1, 1)
        self.decoder_module3 = BasicConv2d(inplanes*10, inplanes*4, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(inplanes*5, inplanes*2, 3, 1, 1)
        self.hcm = HCM(cur_channel = inplanes*2)

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats
        change1_h, change1_w = fa1.size(2), fa1.size(3)

        [fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4] = self.drop([fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4])

        change1 = self.stage1_Conv1(torch.cat([fa1, fb1], 1))
        change2 = self.stage2_Conv1(torch.cat([fa2, fb2], 1))
        change3 = self.stage3_Conv1(torch.cat([fa3, fb3], 1))
        change4 = self.stage4_Conv1(torch.cat([fa4, fb4], 1))

        if self.expand_field is not None:
            change4 = self.expand_field(change4)

        change4_1 = F.interpolate(change4, change1.size()[2:], mode='bilinear', align_corners=True)
        feature_fuse = change4_1 

        change_map = self.decoder(feature_fuse) 
        change4 = self.cgm_4(change4, change_map)
        feature4 = self.decoder_module4(torch.cat([self.upsample2x(change4), change3], 1))
        change3 = self.cgm_3(feature4, change_map)
        feature3 = self.decoder_module3(torch.cat([self.upsample2x(change3), change2], 1))
        change2 = self.cgm_2(feature3, change_map)
        change1 = self.decoder_module2(torch.cat([self.upsample2x(change2), change1], 1))

        change_map = F.interpolate(change_map, (224,224), mode='bilinear', align_corners=True)

        change1 = self.hcm(change1)  

        final_map = self.decoder_final(change1)
        final_map = F.interpolate(final_map,  (224,224), mode='bilinear', align_corners=True)

        return final_map
             
