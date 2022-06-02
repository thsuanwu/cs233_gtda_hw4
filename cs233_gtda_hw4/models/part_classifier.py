"""
Point-Net.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class part_classifier(nn.Module):
    def __init__(self, init_feat_dim=131, n_parts=4, student_optional_hyper_param=None):
        """
        Students:
        You can make a generic function that instantiates a point-net with arbitrary hyper-parameters,
        or go for an implemetnations working only with the hyper-params of the HW.
        Do not use batch-norm, drop-out and other not requested features.
        Just nn.Linear/Conv1D/ReLUs and the (max) poolings.
        
        :param init_feat_dim: input point dimensionality (default 3 for xyz)
        :param conv_dims: output point dimensionality of each layer
        """ #batch_size, num_channels, num_points
        super(part_classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(init_feat_dim, 64, 1),
            nn.Conv1d(64, n_parts, 1),
        )

    def forward(self, x):
        """
        Run forward pass of the PointNet model on a given point cloud.
        :param pointclouds: (B x N+3) point cloud
        """
        x = self.layers(x)
        return x