import torch
import torch.nn as nn

from model.HolisticAttention import HA
#from model.vgg import B2_VGG
from model.vgg_s import vgg16
############################feature extraction block####################################
class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2,x3), 1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        return x
########################################################################################
######################## aggragation three level depth features#########################
class decoder_d(nn.Module):
    def __init__(self, channel):
        super(decoder_d, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat4 = nn.Conv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat5 = nn.Conv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5_1 = nn.Conv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5_2 = nn.Conv2d(3*channel, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x3, x4, x5):
        # x3: 1/16 x4: 1/8 x5: 1/4
        x3_1 = x3
        x4_1 = x4
        x5_1 = x5
        x4_2 = torch.cat((x4_1, self.conv_upsample4(self.upsample(x5_1))), 1)
        x4_2 = self.conv_concat4(x4_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x4_2))), 1)
        x3_2 = self.conv_concat5(x3_2)
        x = self.conv5_1(x3_2)
        x = self.conv5_2(x)
        return x
########################################################################################
###################################boundary decoder ####################################
class decoder_b(nn.Module):
    def __init__(self, channel):
        super(decoder_b, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #
        self.conv_1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_5 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_cat1 = nn.Conv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_cat2 = nn.Conv2d(3*channel, 3*channel, 3, padding=1)
        self.conv3_1 = nn.Conv2d(3*channel, 3*channel, 3, padding=1)
        self.conv3_2 = nn.Conv2d(3*channel, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
    def forward(self, x3, x4, x5):
        # x1: 1 x2: 1/2  x3: 1/4
        x5_1 = x5
        x4_1 = x4
        x3_1 = x3
        x4_1 = self.conv_1(self.upsample(x5_1)) + x4_1
        x3_1 = self.conv_2(self.upsample(self.upsample(x5_1)))+ self.conv_3(self.upsample(x4_1)) + x3_1
        x4_2 = self.conv_cat1(torch.cat((x4_1, self.conv_4(self.upsample(x5_1))), 1))
        x3_2 = self.conv_cat2(torch.cat((x3_1, self.conv_5(self.upsample(x4_2))), 1))
        x = self.conv3_1(x3_2)
        x = self.conv3_2(x)
        return x

########################################################################################
class AttentionLayer(nn.Module):
    def __init__(self, channel, reduction=2, multiply=True):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
                )
        self.multiply = multiply
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        if self.multiply == True:
            return x * y
        else:
            return y
######################## aggragation three level saliency features######################
class decoder_s(nn.Module):
    def __init__(self, channel):
        super(decoder_s, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #
        self.conv_upsample1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample6 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample8 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample9 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_upsample10 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample11 = nn.Conv2d(2 *channel, 2 *channel, 3, padding=1)
        self.conv_upsample12 = nn.Conv2d(channel, channel, 3, padding=1)
        #
        self.channel_att5 = AttentionLayer(channel,reduction=2)
        self.channel_att4 = AttentionLayer(channel,reduction=2)
        self.channel_att3 = AttentionLayer(channel,reduction=2)
        self.channel_ratt5 = AttentionLayer(channel, reduction=2)
        self.channel_ratt4 = AttentionLayer(channel, reduction=2)
        self.channel_ratt3 = AttentionLayer(channel, reduction=2)
        self.channel_reatt5 = AttentionLayer(channel, reduction=2)
        self.channel_reatt4 = AttentionLayer(channel, reduction=2)
        self.channel_reatt3 = AttentionLayer(channel, reduction=2)
        self.channel_rdatt5 = AttentionLayer(channel, reduction=2)
        self.channel_rdatt4 = AttentionLayer(channel, reduction=2)
        self.channel_rdatt3 = AttentionLayer(channel, reduction=2)
        #
        self.conv_concat3 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat4 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat5 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat6 = nn.Conv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv_concat7 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat8 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat9 = nn.Conv2d(2 * channel, channel, 1)
        self.conv_concat10 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat11 = nn.Conv2d(2 * channel, channel, 1)
        self.conv_concat12 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat13 = nn.Conv2d(2 * channel, channel, 1)
        self.conv_concat14 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat15 = nn.Conv2d(2 * channel, channel, 1)
        self.conv_concat16 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat17 = nn.Conv2d(2 * channel, channel, 1)
        self.conv_concat18 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat19 = nn.Conv2d(2 * channel, channel, 1)
        self.conv_concat20 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat21 = nn.Conv2d(2 * channel, channel, 1)
        #
        self.conv5_1 = nn.Conv2d(1* channel, 1*channel, 3, padding=1)
        self.conv5_2 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv5_r = nn.Conv2d(1* channel, 1*channel, 3, padding=1)
        self.conv5_3 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv5_4 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv5_5 = nn.Conv2d(1 * channel, 1, 1)
        self.conv5_6 = nn.Conv2d(1 * channel, 1 * channel, 3 , padding=1)
        self.conv5_7 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv5_8 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv5_9 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv5_be = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv5_bd = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv4_be = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv4_bd = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_be = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_bd = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.convcat5 = nn.Conv2d(2 * channel, 1 * channel, 3, padding=1)
        self.convadd5 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.convcat52 = nn.Conv2d(2 * channel, 1 * channel, 3, padding=1)
        self.convcat53 = nn.Conv2d(2 * channel, 1 * channel, 3, padding=1)
        self.convcat54 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.convadd52 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.convcat4 = nn.Conv2d(2 * channel, 1 * channel, 3, padding=1)
        self.convadd4 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.convcat42 = nn.Conv2d(2 * channel, 1 * channel, 3, padding=1)
        self.convcat43 = nn.Conv2d(2 * channel, 1 * channel, 3, padding=1)
        self.convcat44 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.convadd42 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.convcat32 = nn.Conv2d(2 * channel, 1 * channel, 3, padding=1)
        self.convcat33 = nn.Conv2d(2 * channel, 1 * channel, 3, padding=1)
        self.convcat34 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.convadd32 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.convcat3 = nn.Conv2d(2 * channel, 1 * channel, 3, padding=1)
        self.convadd3 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv5_res1 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv5_res2 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv5_red1 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv5_red2 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)

        self.conv4_1 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv4_2 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv4_3 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv4_4 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv4_8 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv4_9 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv4_r = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv4_res1 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv4_res2 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv4_red1 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv4_red2 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv3_2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv3_3 = nn.Conv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv3_4 = nn.Conv2d(3 * channel, 1 * channel, 3, padding=1)
        self.conv3_5 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_6 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_7 = nn.Conv2d(1 * channel, 1, 1)
        self.conv3_8 = nn.Conv2d(1 * channel, 1, 3, padding=1)
        self.conv3_9 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_10 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_11 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_12 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_13 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_14 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_res1 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_res2 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_red1 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_red2 = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv3_r = nn.Conv2d(1 * channel, 1 * channel, 3, padding=1)
        self.conv2_1 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv2_2 = nn.Conv2d(2 * channel, 1, 3, padding=1)
        self.relu5_1 = nn.ReLU(True)
        self.relu5_2 = nn.ReLU(True)
        self.relu5_6 = nn.ReLU(True)
        self.relu5_7 = nn.ReLU(True)
        self.relu5_8 = nn.ReLU(True)
        self.relu5_9 = nn.ReLU(True)
        self.relu5_r = nn.ReLU(True)
        self.relu4_1 = nn.ReLU(True)
        self.relu4_2 = nn.ReLU(True)
        self.relu4_3 = nn.ReLU(True)
        self.relu4_4 = nn.ReLU(True)
        self.relu4_8 = nn.ReLU(True)
        self.relu4_9 = nn.ReLU(True)
        self.relu4_r = nn.ReLU(True)
        self.relu3_1 = nn.ReLU(True)
        self.relu3_2 = nn.ReLU(True)
        self.relu3_3 = nn.ReLU(True)
        self.relu3_4 = nn.ReLU(True)
        self.relu3_5 = nn.ReLU(True)
        self.relu3_6 = nn.ReLU(True)
        self.relu3_7 = nn.ReLU(True)
        self.relu3_8 = nn.ReLU(True)
        self.relu3_9 = nn.ReLU(True)
        self.relu3_10 = nn.ReLU(True)
        self.relu3_13 = nn.ReLU(True)
        self.relu3_14 = nn.ReLU(True)
        self.relu3_r = nn.ReLU(True)
        self.pool2_1 = nn.AvgPool2d(2,stride=2)
        self.pool4_1 = nn.AvgPool2d(2, stride=2)
        self.pool4_2 = nn.AvgPool2d(2, stride=2)
        self.pool5_1 = nn.AvgPool2d(2, stride=2)
        self.pool5_2 = nn.AvgPool2d(2, stride=2)
        self.pool5_4 = nn.AvgPool2d(2, stride=2)
        self.pool5_3 = nn.AvgPool2d(2, stride=2)
        self.pool5_5 = nn.AvgPool2d(2, stride=2)
        self.pool3_1 = nn.AvgPool2d(2, stride=2)
        self.pool3_2 = nn.AvgPool2d(2, stride=2)
        self.maxpool5 = nn.MaxPool2d(2,stride = 2)
        self.maxpool4 = nn.MaxPool2d(4, stride=4)
        self.maxpool3 = nn.MaxPool2d(4, stride=4)
        self.pool_depth = nn.AvgPool2d(2, stride=2)
        self.pool_depth2 = nn.AvgPool2d(2, stride=2)
        self.sigmoid3_1 =nn.Sigmoid()
        self.sigmoid5_1 = nn.Sigmoid()

        self.HA = HA()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x3, x4 ,x5, x3_2, x4_2,x5_2,x3_3, x4_3, x5_3,depth_):
        #  x3: 1/4 x4:1/8 x5:1/16
        depth_ = self.pool_depth(self.pool_depth2(depth_))
        x5_s_1 = x5
        x5_s_1 = self.relu5_8(self.conv5_8(x5_s_1))
        x5_s_1 = self.relu5_9(self.conv5_9(x5_s_1))
        x5_d = x5_3
        x5_d = self.relu5_1(self.conv5_1(x5_d))
        x5_c = self.relu5_2(self.conv5_2(x5_d))
        x5_b = x5_2
        x5_b = self.relu5_6(self.conv5_6(x5_b))
        x5_b = self.relu5_7(self.conv5_7(x5_b))
        x5_sal = self.conv5_4(self.conv5_3(x5_s_1))
        x5_sal = self.conv5_4(x5_sal)
        x_att5= self.conv5_5(x5_sal)
        x_att5 = self.sigmoid5_1(x_att5)
        n_, _, _, _ = x_att5.size()
        res_f = torch.zeros((n_, 1, 64, 64))
        res_dsf = torch.zeros((n_, 1, 64, 64))
        for jj in range(10):
            res = depth_ * 255
            target = self.upsample(self.upsample(x_att5))
            target = target * 255
            res1 = (res >= (255.0 / 10) * jj)
            res1 = res1.type(torch.FloatTensor)
            res1 = res1.cuda()
            res3 = (res <= (255.0 / 10) * (jj + 1))
            res3 = res3.type(torch.FloatTensor)
            res3 = res3.cuda()
            res2 = res * res1 * res3
            res2[res2 > 0] = 255
            res_sim = res2 * (target / 255)
            res_bi = res2
            res_res = res2
            total = target.mean(dim=3)
            total = total.mean(dim=2)
            total_d = res_bi.mean(dim=3)
            total_d = total_d.mean(dim=2)
            res_sim = res_sim.mean(dim = 3)
            res_sim = res_sim.mean(dim=2 )
            weight = torch.div(res_sim,total)
            weight = torch.unsqueeze(weight, -1)
            weight = torch.unsqueeze(weight, -1)
            weight_d1 = torch.div(res_sim, total_d + 1e-4)
            weight_d = torch.unsqueeze(weight_d1, -1)
            weight_d = torch.unsqueeze(weight_d, -1)
            weight_d = (  weight_d) * (weight_d/(weight_d + 1e-4))
            res_f = res_f.cuda()
            res_dsf = res_dsf.cuda()
            res__ = torch.mul(res_res , weight)
            res_dsf2 = torch.mul(res_res, weight_d)
            res_f = res_f + res__
            res_dsf = res_dsf + res_dsf2
        res_f = res_f / 255
        res_dsf = res_dsf / 255
        res_df5 = self.pool5_3(self.pool5_4(res_f))
        res_f5 = self.pool5_2(self.pool5_1(res_dsf))
        x5_reatt = self.upsample(self.maxpool5( x_att5))
        x5_res2 = x5_s_1 * x5_reatt
        x5_res2 = self.channel_reatt5(x5_res2)
        x5_res = x5_s_1 * res_f5
        x5_ratt = self.channel_ratt5(x5_res)
        x5_ratt = self.conv5_res1(x5_ratt)
        x5_ratt = self.conv5_res2(x5_ratt)
        x5_res = x5_ratt
        x5_stru = x5_s_1 * x5_b
        x5_stru = self.conv5_be(x5_stru)
        x5_stru = self.relu(x5_stru + x5_s_1)
        x5_rf = torch.cat((x5_res2 , x5_res),1)
        x5_rf1 = self.convcat5(x5_rf)
        x5_rf = x5_rf1 + x5_stru
        x5_rf = self.convadd5(x5_rf)
        x5_red = x5_c * x_att5
        x5_att = self.channel_att5(x5_red)
        x5_att = self.conv5_red1(x5_att)
        x5_att = self.conv5_red2(x5_att)
        x5_red2 = x5_c * res_df5
        x5_red2 = self.channel_rdatt5(x5_red2)
        x5_strud = x5_c * x5_b
        x5_strud = self.conv5_bd(x5_strud)
        x5_strud = self.relu(x5_strud + x5_c)
        x5_df = torch.cat((x5_att , x5_red2),1)
        x5_df = self.convcat52(x5_df)
        x5_df = x5_df + x5_strud
        x5_df = self.convadd52(x5_df)
        x5_f = torch.cat((x5_rf , x5_df),1)
        x5_f = self.convcat53(x5_f)
        x5_f = self.convcat54(x5_f)
        x5_s_r = self.conv5_r(x5_f)
        x5_s_r = self.relu5_r(x5_s_r)
        x5_s = x5_f + x5_s_r
        x4_s_1 = x4
        x4_s_1 = self.conv4_8(x4_s_1)
        x4_s_1 = self.relu4_8(x4_s_1)
        x4_s_1 = self.conv4_9(x4_s_1)
        x4_s_1 = self.relu4_9(x4_s_1)
        x4_d = x4_3
        x4_d = self.conv4_1(x4_d)
        x4_d = self.relu4_1(x4_d)
        x4_c = self.conv4_2(x4_d)
        x4_c = self.relu4_2(x4_c)
        x4_b = x4_2
        x4_b = self.conv4_3(x4_b)
        x4_b = self.relu4_3(x4_b)
        x4_b = self.conv4_4(x4_b)
        x4_b = self.relu4_4(x4_b)
        x4_reatt = self.upsample(self.upsample((self.maxpool4( self.upsample(1-x_att5)))))
        x4_res2 = x4_s_1 * x4_reatt
        x4_res2 = self.channel_reatt4(x4_res2)
        x4_res = x4_s_1 * self.pool4_1(res_dsf)
        x4_res = self.channel_ratt4(x4_res)
        x4_res = self.conv4_res1(x4_res)
        x4_res = self.conv4_res2(x4_res)
        x4_stru = x4_s_1 * x4_b
        x4_stru = self.conv4_be(x4_stru)
        x4_stru = self.relu(x4_stru + x4_s_1)
        x4_rf = torch.cat((x4_res2, x4_res), 1)
        x4_rf = self.convcat4(x4_rf)
        x4_rf = x4_rf + x4_stru
        x4_rf = self.convadd4(x4_rf)
        x4_red = x4_c * self.upsample(x_att5)
        x4_att = self.channel_att4(x4_red)
        x4_att = self.conv4_red1(x4_att)
        x4_att = self.conv4_red2(x4_att)
        x4_red2 = x4_c * self.pool4_2(res_f)
        x4_red2 = self.channel_rdatt4(x4_red2)
        x4_strud = x4_c * x4_b
        x4_strud = self.conv4_bd(x4_strud)
        x4_strud = self.relu(x4_strud + x4_c)
        x4_df = torch.cat((x4_att, x4_red2), 1)
        x4_df = self.convcat42(x4_df)
        x4_df = x4_df + x4_strud
        x4_df = self.convadd42(x4_df)
        x4_f = torch.cat((x4_rf, x4_df), 1)
        x4_f = self.convcat43(x4_f)
        x4_f = self.convcat44(x4_f)
        x4_s_r = self.conv4_r(x4_f)
        x4_s_r = self.relu4_r(x4_s_r)
        x4_s = x4_f + x4_s_r
        x3_s_1 = x3
        x3_s_1 = self.conv3_13(x3_s_1)
        x3_s_1 = self.relu3_13(x3_s_1)
        x3_s_1 = self.conv3_14(x3_s_1)
        x3_s_1 = self.relu3_14(x3_s_1)
        x3_d = x3_3
        x3_d = self.conv3_9(x3_d)
        x3_d = self.relu3_7(x3_d)
        x3_d = self.conv3_10(x3_d)
        x3_c = self.relu3_8(x3_d)
        x3_b = x3_2
        x3_b = self.conv3_11(x3_b)
        x3_b = self.relu3_9(x3_b)
        x3_b = self.conv3_12(x3_b)
        x3_b = self.relu3_10(x3_b)
        x3_red = x3_c * self.upsample(self.upsample(x_att5))
        x3_att = self.channel_att3(x3_red)
        x3_att = self.conv3_red1(x3_att)
        x3_att = self.conv3_red2(x3_att)
        x3_res = x3_s_1 * res_dsf
        x3_res = self.channel_ratt3(x3_res)
        x3_res = self.conv3_res1(x3_res)
        x3_res = self.conv3_res2(x3_res)
        x3_reatt = self.upsample(self.upsample(self.maxpool3((self.upsample(self.upsample(1-x_att5))))))
        x3_res2 = x3_s_1 * x3_reatt
        x3_res2 = self.channel_reatt3(x3_res2)
        x3_stru = x3_s_1 * x3_b
        x3_stru = self.conv3_be(x3_stru)
        x3_stru = self.relu(x3_stru + x3_s_1)
        x3_rf = torch.cat((x3_res2, x3_res), 1)
        x3_rf = self.convcat3(x3_rf)
        x3_rf = x3_rf + x3_stru
        x3_rf = self.convadd3(x3_rf)
        x3_red2 = x3_c * res_f
        x3_red2 = self.channel_rdatt3(x3_red2)
        x3_strud = x3_c * x3_b
        x3_strud = self.conv3_bd(x3_strud)
        x3_strud = self.relu(x3_strud + x3_c)
        # fuse feature
        x3_df = torch.cat((x3_att, x3_red2), 1)
        x3_df = self.convcat32(x3_df)
        x3_df = x3_df + x3_strud
        x3_df = self.convadd32(x3_df)
        x3_f = torch.cat((x3_rf, x3_df), 1)
        x3_f = self.convcat33(x3_f)
        x3_f = self.convcat34(x3_f)
        x3_s_r = self.conv3_r(x3_f)
        x3_s_r = self.relu3_r(x3_s_r)
        x3_s = x3_f + x3_s_r
        x3_s = x3_s + self.conv_upsample6(self.upsample(x4_s)) + self.conv_upsample7(self.upsample(self.upsample(x5_s)))
        x4_s = x4_s + self.conv_upsample12(self.upsample(x5_s))
        x4_s_2 = torch.cat((x4_s, self.conv_upsample8(self.upsample(x5_s))), 1)
        x4_s_2 = self.conv_concat5(x4_s_2)
        x3_s_2 = torch.cat((x3_s, self.conv_upsample9(self.upsample(x4_s_2))), 1)
        x3_s_2 = self.conv_concat6(x3_s_2)
        x3_s_2 = self.conv3_3(x3_s_2)
        x3_s_2 = self.conv3_4(x3_s_2)
        x_attention = self.conv3_8(x3_s_2)
        return x_attention,x_att5

class CPD_VGG(nn.Module):
    def __init__(self, channel=32):
        super(CPD_VGG, self).__init__()
        self.vgg = VGG_Pr()
        self.vgg_d = VGG_Pr()
        self.macb3_1 = RFB(256, channel)
        self.macb4_1 = RFB(512, channel)
        self.macb5_1 = RFB(512, channel)
        self.agg1 = decoder_d(channel)
        self.macb3_2 = RFB(256, channel)
        self.macb4_2 = RFB(512, channel)
        self.macb5_2 = RFB(512, channel)
        self.agg2 = decoder_s(channel)
        self.macb3_3 = RFB(256, channel)
        self.macb4_3 = RFB(512, channel)
        self.macb5_3 = RFB(512, channel)
        self.agg3 = decoder_b(channel)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x,x_d,depth_):
        #depth VGG_Stream
        x1 = self.vgg_d.conv1(x_d)
        x2 = self.vgg_d.conv2(x1)
        x3 = self.vgg_d.conv3(x2)
        x4 = self.vgg_d.conv4_1(x3)
        x5 = self.vgg_d.conv5_1(x4)
        x3_d = self.macb3_1(x3)
        x4_d = self.macb4_1(x4)
        x5_d = self.macb5_1(x5)
        saliency_d = self.agg1(x3_d,x4_d,x5_d)
        # RGB VGG_Stream
        x1_s = self.vgg.conv1(x)
        x2_s = self.vgg.conv2(x1_s)
        x3_s = self.vgg.conv3(x2_s)
        x4_s = self.vgg.conv4_1(x3_s)
        x5_s = self.vgg.conv5_1(x4_s)
        x3_s1 = self.macb3_2(x3_s)
        x4_s1 = self.macb4_2(x4_s)
        x5_s1 = self.macb5_2(x5_s)
        x3_b = self.macb3_3(x3_s)
        x4_b = self.macb4_3(x4_s)
        x5_b = self.macb5_3(x5_s)
        boundary = self.agg3(x3_b, x4_b, x5_b)
        detection,x_att5= self.agg2(x3_s1, x4_s1, x5_s1,x3_b,x4_b,x5_b,x3_d.detach(),x4_d.detach(),x5_d.detach(),depth_)



        return self.upsample1(saliency_d), self.upsample1(detection), self.upsample1(boundary),self.upsample1(self.upsample1(x_att5))