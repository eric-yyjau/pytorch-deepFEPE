""" down level of deepF network, refer to deepFNet for top level
Reviewed and tested by You-Yi on 07/13/2020.

Reference:
    deepF, learning to find good correspondences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.nn.functional import grid_sample

class ErrorEstimator(nn.Module):
    def __init__(self, input_size, output_size=1, if_bn=False):
        super(ErrorEstimator, self).__init__()

        print('ErrorEstimator@ErrorEstimators.py')
        inplace = True
        hasbias = True
        learn_affine = True
        if if_bn:
            self.fw = nn.Sequential(
                # nn.InstanceNorm1d(input_size),
                nn.Conv1d(input_size, 64, kernel_size=1, bias=hasbias),
                nn.BatchNorm1d(64),
                nn.InstanceNorm1d(64, affine=learn_affine),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv1d(64,128, kernel_size=1, bias=hasbias),
                nn.BatchNorm1d(128),
                nn.InstanceNorm1d(128, affine=learn_affine),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv1d(128,1024,kernel_size=1, bias=hasbias),
                nn.BatchNorm1d(1024),
                nn.InstanceNorm1d(1024, affine=learn_affine),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv1d(1024,512, kernel_size=1, bias=hasbias),
                nn.BatchNorm1d(512),
                nn.InstanceNorm1d(512, affine=learn_affine),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv1d(512,256, kernel_size=1, bias=hasbias),
                nn.BatchNorm1d(256),
                nn.InstanceNorm1d(256, affine=learn_affine),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv1d(256, output_size, kernel_size=1, bias=False))
        else:
            self.fw = nn.Sequential(
                # nn.InstanceNorm1d(input_size),
                nn.Conv1d(input_size, 64, kernel_size=1, bias=hasbias),
                nn.InstanceNorm1d(64, affine=learn_affine),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv1d(64,128, kernel_size=1, bias=hasbias),
                nn.InstanceNorm1d(128, affine=learn_affine),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv1d(128,1024,kernel_size=1, bias=hasbias),
                nn.InstanceNorm1d(1024, affine=learn_affine),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv1d(1024,512, kernel_size=1, bias=hasbias),
                nn.InstanceNorm1d(512, affine=learn_affine),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv1d(512,256, kernel_size=1, bias=hasbias),
                nn.InstanceNorm1d(256, affine=learn_affine),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv1d(256, output_size, kernel_size=1, bias=hasbias))

    def forward(self, data):
        # print('ErrorEstimator')
        return self.fw(data)


class ErrorEstimatorDes(nn.Module):
    def __init__(self, input_size, des_size):
        super(ErrorEstimatorDes, self).__init__()

        inplace = True
        hasbias = True
        learn_affine = True
        self.fw_data_out = nn.Sequential(
            nn.Conv1d(2048,512, kernel_size=1, bias=hasbias),
            nn.InstanceNorm1d(512, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(512,256, kernel_size=1, bias=hasbias),
            nn.InstanceNorm1d(256, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(256, 1, kernel_size=1, bias=hasbias))
        self.fw_data_head = nn.Sequential(
            # nn.InstanceNorm1d(input_size),
            nn.Conv1d(input_size, 64, kernel_size=1, bias=hasbias),
            nn.InstanceNorm1d(64, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(64,128, kernel_size=1, bias=hasbias),
            nn.InstanceNorm1d(128, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(128,1024,kernel_size=1, bias=hasbias),
            nn.InstanceNorm1d(1024, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace))
        self.fw_des_head = nn.Sequential(
            # nn.InstanceNorm1d(input_size),
            nn.Conv1d(des_size, 256, kernel_size=1, bias=hasbias),
            nn.InstanceNorm1d(256, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(256,512, kernel_size=1, bias=hasbias),
            nn.InstanceNorm1d(512, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace))

    def forward(self, data, des1, des2):
        # print('ErrorEstimatorDes')
        data_head_feat = self.fw_data_head(data)

        des1_head_feat = self.fw_des_head(des1)
        des2_head_feat = self.fw_des_head(des2)

        feat_all = torch.cat((data_head_feat, des1_head_feat, des2_head_feat), 1)

        return self.fw_data_out(feat_all)


class ErrorEstimatorFeatFusion(nn.Module):
    def __init__(self, points_channel, feat_channel, output_size=1):
        super(ErrorEstimatorFeatFusion, self).__init__()
        print('ErrorEstimatorFeatFusion@ErrorEstimators.py')
        inplace = True
        hasbias = True
        learn_affine = True

        self.l1 = nn.Sequential(
            # nn.InstanceNorm1d(input_size),
            nn.Conv1d(points_channel+feat_channel, 64, kernel_size=1, bias=hasbias),
            nn.BatchNorm1d(64),
            nn.InstanceNorm1d(64, affine=learn_affine),
            nn.ReLU(inplace=inplace))
        self.l2 = nn.Sequential(
            nn.Conv1d(64,128, kernel_size=1, bias=hasbias),
            nn.BatchNorm1d(128),
            nn.InstanceNorm1d(128, affine=learn_affine),
            nn.ReLU(inplace=inplace))
        self.l3 = nn.Sequential(
            nn.Conv1d(128,1024,kernel_size=1, bias=hasbias),
            nn.BatchNorm1d(1024),
            nn.InstanceNorm1d(1024, affine=learn_affine),
            nn.ReLU(inplace=inplace))
        self.l4 = nn.Sequential(
            nn.Conv1d(1024,512, kernel_size=1, bias=hasbias),
            nn.BatchNorm1d(512),
            nn.InstanceNorm1d(512, affine=learn_affine),
            nn.ReLU(inplace=inplace))
        self.l5 = nn.Sequential(
            nn.Conv1d(512,256, kernel_size=1, bias=hasbias),
            nn.BatchNorm1d(256),
            nn.InstanceNorm1d(256, affine=learn_affine),
            nn.ReLU(inplace=inplace),
            nn.Conv1d(256, output_size, kernel_size=1, bias=hasbias))

    def forward(self, pts_in, des_in):
        # print('ErrorEstimator')
        tensor_in = torch.cat((pts_in, des_in), 1)
        # print(tensor_in.size(), pts_in.size(), des_in.size())
        feat1 = self.l1(tensor_in)
        # print(feat1.size())
        feat2 = self.l2(feat1)
        # print(feat2.size())
        feat3 = self.l3(feat2)
        # print(feat3.size())
        feat4 = self.l4(feat3)
        # print(feat4.size())
        feat5 = self.l5(feat4)
        # print(feat5.size())

        return feat5

class ErrorEstimatorFeatFusion2Head(nn.Module):
    def __init__(self, points_channel, feat_channel):
        super(ErrorEstimatorFeatFusion2Head, self).__init__()
        print('ErrorEstimatorFeatFusion2Head@ErrorEstimators.py')
        inplace = True
        hasbias = True
        learn_affine = True

        self.l1_1 = nn.Sequential(
            # nn.InstanceNorm1d(input_size),
            nn.Conv1d(points_channel, 64, kernel_size=1, bias=hasbias),
            # nn.BatchNorm1d(64),
            nn.InstanceNorm1d(64, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace))
        self.l2_1 = nn.Sequential(
            nn.Conv1d(64,128, kernel_size=1, bias=hasbias),
            # nn.BatchNorm1d(128),
            nn.InstanceNorm1d(128, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace))
        self.l3_1 = nn.Sequential(
            nn.Conv1d(128,1024,kernel_size=1, bias=hasbias),
            # nn.BatchNorm1d(1024),
            nn.InstanceNorm1d(1024, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace))

        self.l1_2 = nn.Sequential(
            # nn.InstanceNorm1d(input_size),
            nn.Conv1d(feat_channel, 64, kernel_size=1, bias=hasbias),
            # nn.BatchNorm1d(64),
            # nn.InstanceNorm1d(64, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace))
        self.l2_2 = nn.Sequential(
            nn.Conv1d(64,128, kernel_size=1, bias=hasbias),
            # nn.BatchNorm1d(128),
            # nn.InstanceNorm1d(128, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace))
        self.l3_2 = nn.Sequential(
            nn.Conv1d(128,1024,kernel_size=1, bias=hasbias),
            # nn.BatchNorm1d(1024),
            # nn.InstanceNorm1d(1024, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace))

        self.l4 = nn.Sequential(
            nn.Conv1d(2048,512, kernel_size=1, bias=hasbias),
            # nn.BatchNorm1d(512),
            nn.InstanceNorm1d(512, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace))
        self.l5 = nn.Sequential(
            nn.Conv1d(512,256, kernel_size=1, bias=hasbias),
            # nn.BatchNorm1d(256),
            nn.InstanceNorm1d(256, affine=learn_affine),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(256, 1, kernel_size=1, bias=hasbias))

    def forward(self, pts_in, des_in):
        # print('ErrorEstimator')
        # tensor_in = torch.cat((pts_in, des_in), 1)
        feat1_1 = self.l1_1(pts_in)
        feat2_1 = self.l2_1(feat1_1)
        feat3_1 = self.l3_1(feat2_1)
        feat1_2 = self.l1_2(des_in)
        feat2_2 = self.l2_2(feat1_2)
        feat3_2 = self.l3_2(feat2_2)
        feat4 = self.l4(torch.cat((feat3_1, feat3_2), 1))
        feat5 = self.l5(feat4)

        return feat5



def main():
    model_params = {
        "input_size": 4,
        "output_size": 1,
        "if_bn": False,
    }
    net = ErrorEstimator(**model_params)
    print(f"ErrorEstimator net: {net}")
    pass

if __name__ == "__main__":
    main()