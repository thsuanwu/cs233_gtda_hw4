"""
Multi-layer perceptron.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""


from torch import nn


class MLP(nn.Module):
    """ Multi-layer perceptron. That is a k-layer deep network where each layer is a fully-connected (nn.Linear) layer, with
    (optionally) batch-norm, a non-linearity and dropout.

    Students: again, you can use this scaffold to make a generic MLP that can be used with multiple-hyper parameters
    or, opt for a perhaps simpler custom variant that just does so for HW4. For HW4 do not use batch-norm, drop-out
    or other non-requested features, for the non-bonus question.
    """

    def __init__(self, in_feat_dim=128, out_channels=1024, b_norm=False, dropout_rate=0, non_linearity=nn.ReLU(inplace=True)):
        """Constructor
        :param in_feat_dim: input feature dimension
        :param out_channels: list of ints describing each the number hidden/final neurons.
        :param b_norm: True/False, or list of booleans
        :param dropout_rate: int, or list of int values
        :param non_linearity: nn.Module
        """
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Run forward pass of MLP
        :param x: (B x in_feat_dim) point cloud
        """
        x = self.layers(x)
        return x
        
