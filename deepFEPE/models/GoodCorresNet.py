""" Deprecated
Reviewed and tested by You-Yi on 07/13/2020.

Reference:
    learning to find good correspondences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


# Import Shaper (install first: https://github.com/haosulab/shaper)
# If cannot find: export SHAPER_MODELS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils/deepSfm_ori/models/shaper/shaper/models/pointnet'
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# sys.path.append(os.environ['SHAPER_MODELS_PATH']+'/pointnet') # temparily disabled
# sys.path.append('/home/yyjau/Documents/deepSfm/packages/pointnet/part_seg') # package for youyi

# from pointnet_part_seg import *

import deepFEPE.dsac_tools.utils_F as utils_F # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils' 

def get_E_ests(x1, x2, Ks, logits_weights, if_normzliedK=True):
    # E_ests_list = []
    # for x1_single, x2_single, K, w in zip(x1, x2, Ks, logits_weights):
    #     E_est = utils_F._E_from_XY(x1_single, x2_single, K, torch.diag(w), if_normzliedK=if_normzliedK)
    #     E_ests_list.append(E_est)
    # E_ests = torch.stack(E_ests_list)
    E_ests = utils_F._E_from_XY_batch(x1, x2, Ks, torch.diag_embed(logits_weights, dim1=-2, dim2=-1), if_normzliedK=if_normzliedK)
    return E_ests


class GoodCorresNet(nn.Module):
    """PointNet for part segmentation
    References:
        https://github.com/charlesq34/pointnet/blob/master/part_seg/pointnet_part_seg.py
    """

    def __init__(self,
                 in_channels,
                 # num_classes,
                 num_seg_classes=1,
                 stem_channels=(64, 128, 128),
                 local_channels=(512, 2048),
                 cls_channels=(256, 256),
                 seg_channels=(256, 256, 128),
                 dropout_prob_cls=0.3,
                 dropout_prob_seg=0.2,
                 with_instance_norm=True,
                 with_transform=False,
                 bn=False):
        """
        Args:
           in_channels (int): the number of input channels
           out_channels (int): the number of output channels
           stem_channels (tuple of int): the numbers of channels in stem feature extractor
           local_channels (tuple of int): the numbers of channels in local mlp
           cls_channels (tuple of int): the numbers of channels in classification mlp
           seg_channels (tuple of int): the numbers of channels in segmentation mlp
           dropout_prob_cls (float): the probability to dropout in classification mlp
           dropout_prob_seg (float): the probability to dropout in segmentation mlp
           with_transform (bool): whether to use TNet to transform features.
        """
        super(GoodCorresNet, self).__init__()
        print('Created GoodCorresNet!!!')

        self.in_channels = in_channels
        # self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.with_transform = with_transform

        bn=False
        # stem
        self.stem = Stem(in_channels, stem_channels, with_transform=with_transform, with_instance_norm=with_instance_norm, bn=bn)
        self.mlp_local = SharedMLP(stem_channels[-1], local_channels, with_instance_norm=with_instance_norm, bn=bn)

        # # classification
        # # Notice that we apply dropout to each classification mlp.
        # self.mlp_cls = MLP(local_channels[-1], cls_channels, dropout=dropout_prob_cls)
        # self.cls_logit = nn.Linear(cls_channels[-1], num_classes, bias=True)

        # part segmentation
        # Notice that the original repo concatenates global feature, one hot class embedding,
        # stem features and local features. However, the paper does not use last local feature.
        # Here, we follow the released repo.

        # in_channels_seg = local_channels[-1] + num_classes + sum(stem_channels) + sum(local_channels)
        in_channels_seg = local_channels[-1] + sum(stem_channels) + sum(local_channels)
        self.mlp_seg = SharedMLP(in_channels_seg, seg_channels[:-1], dropout_prob=dropout_prob_seg, with_instance_norm=with_instance_norm, bn=bn)
        self.conv_seg = Conv1d(seg_channels[-2], seg_channels[-1], 1, with_instance_norm=with_instance_norm, bn=bn)
        self.logits = nn.Conv1d(seg_channels[-1], num_seg_classes, 1, bias=True)

        # self.init_weights()

    def forward(self, data_batch):
        # print(data_batch['matches_xy_ori'].size())
        # x = data_batch['matches_xy_ori'].transpose(1, 2) # data_batch['matches_xy']: [batch_size, N, D]
        # cls_label = data_batch["cls_label"]
        x = data_batch # [batch_size, D, N]

        num_points = x.shape[2]
        end_points = {}

        # stem
        stem_feature, end_points_stem = self.stem(x)
        if self.with_transform:
            end_points["trans_input"] = end_points_stem["trans_input"]
            end_points["trans_feature"] = end_points_stem["trans_feature"]
        stem_features = end_points_stem["stem_features"]

        # mlp for local features
        local_features = []
        x = stem_feature
        for ind, mlp in enumerate(self.mlp_local):
            x = mlp(x)
            local_features.append(x)

        # max pool over points
        global_feature, max_indices = torch.max(x, 2)  # (batch_size, local_channels[-1])
        end_points['key_point_inds'] = max_indices

        # classification
        # x = global_feature
        # x = self.mlp_cls(x)
        # cls_logit = self.cls_logit(x)

        # segmentation
        global_feature_expand = global_feature.unsqueeze(2).expand(-1, -1, num_points) # torch.Size([8, 2048, 1311])
        # with torch.no_grad():
        #     I = torch.eye(self.num_classes, dtype=global_feature.dtype, device=global_feature.device)
        #     one_hot = I[cls_label]  # (batch_size, num_classes)
        #     one_hot_expand = one_hot.unsqueeze(2).expand(-1, -1, num_points) # torch.Size([8, 1, 1311])

        # stem_features: [[8, 64, 1311], [8, 128, 1311], [8, 128, 1311]]
        # local_features: [[8, 512, 1311], [8, 2048, 1311]]
        # x = torch.cat(stem_features + local_features + [global_feature_expand, one_hot_expand], dim=1)
        x = torch.cat(stem_features + local_features + [global_feature_expand], dim=1)
        x = self.mlp_seg(x)
        x = self.conv_seg(x)
        logits = self.logits(x) # No activation

        # preds = {
        #     # "cls_logit": cls_logit,
        #     "logits": logits.squeeze(1) # [batch_size, N]
        # }
        # preds.update(end_points)

        # return preds
        return logits

    def init_weights(self):
        self.mlp_local.init_weights(xavier_uniform)
        # self.mlp_cls.init_weights(xavier_uniform)
        self.mlp_seg.init_weights(xavier_uniform)
        self.conv_seg.init_weights(xavier_uniform)
        # nn.init.xavier_uniform_(self.cls_logit.weight)
        # nn.init.zeros_(self.cls_logit.bias)
        nn.init.xavier_uniform_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)
        # Set batch normalization to 0.01 as default
        set_bn(self, momentum=0.01)


def main():
    model_params = {
        "in_channels": 4,
        "bn": False
        
    }
    net = GoodCorresNet(**model_params)
    print(f"deepF net: {net}")
    pass

if __name__ == "__main__":
    main()