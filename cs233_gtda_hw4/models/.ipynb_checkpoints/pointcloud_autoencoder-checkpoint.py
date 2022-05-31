"""
PC-AE.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
from torch import nn
from ..in_out.utils import AverageMeter
from ..losses.chamfer import chamfer_loss

# In the unlikely case where you cannot use the JIT chamfer implementation (above) you can use the slower
# one that is written in pure pytorch:
#from ..losses.nn_distance import chamfer_loss


class PointcloudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        """ AE constructor.
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        """
        super(PointcloudAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, pointclouds):
        """Forward pass of the AE
            :param pointclouds: B x N x 3
        """
        return self.decoder(self.embed(pointclouds))
        

    def train_for_one_epoch(self, loader, optimizer, device='cuda'):
        """ Train the autoencoder for one epoch based on the Chamfer loss.
        :param loader: (train) pointcloud_dataset loader
        :param optimizer: torch.optimizer
        :param device: cuda? cpu?
        :return: (float), average loss for the epoch.
        """        
        self.train()
        loss_meter = AverageMeter()
        
        ### My Work ###
        
        for data in loader:
            optimizer.zero_grad()
            point_cloud = data['point_cloud'].to(device)
            # Check dims
            #print(point_cloud.shape)
            loss = chamfer_loss(self.reconstruct(point_cloud, device=device), point_cloud).mean()
            loss_meter.update(loss, point_cloud.size(0))
        
        return loss_meter.avg
    
    @torch.no_grad()
    def embed(self, pointclouds):
        """ Extract from the input pointclouds the corresponding latent codes.
        :param pointclouds: B x N x 3
        :return: B x latent-dimension of AE
        """
        return self.encoder(pointclouds)
        

    @torch.no_grad()
    def reconstruct(self, loader, device='cuda'):
        """ Reconstruct the point-clouds via the AE.
        :param loader: pointcloud_dataset loader
        :param device: cpu? cuda?
        :return: Left for students to decide
        """
        x = self.embed((loader.to(device)))
        #print('embed', x.shape)
        y = torch.squeeze(x)
        #print('squeezed', y.shape)
        y = self.decoder(y)
        #print('decoder', y.shape)
        return x