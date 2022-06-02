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
        # Takes pointclouds as input
        """Forward pass of the AE
            :param pointclouds: B x N x 3
        """
        #print('input', pointclouds.shape)
        x = torch.transpose(pointclouds, 1, 2)
        #print('transposed for encoder', x.shape)
        x = self.encoder(x)
        #print('encoded', x.shape)
        #x = torch.transpose(x, 1, 2)
        x = x.squeeze(-1)
        #print('squeezed for decoder', x.shape)
        x = self.decoder(x)
        #print('decoded', x.shape)
        return x
        

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
        
        for batch in loader:
            optimizer.zero_grad()
            
            pointclouds = batch['point_cloud'].to(device)
            loss = chamfer_loss(self.forward(pointclouds), pointclouds).mean()
            #print(loss)
            loss_meter.update(loss, len(loader))
            
            loss.backward()
            optimizer.step()
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
        recons = []
        losses = []
        for batch in loader:
            pointclouds = batch["point_cloud"].to(device)
            recon = self.forward(pointclouds)
            loss = chamfer_loss(recon, pointclouds)
            
            losses.append(loss)
            recons.append(recon)
        return recons, losses
    
    def extract_latent_code(self, loader, device='cuda'):
        latent_codes = []
        test_names = []
        for batch in loader:
            pointclouds = batch["point_cloud"].to(device)
            x = torch.transpose(pointclouds, 1, 2)
            latent = self.embed(x)
            latent = latent.squeeze(-1)
            print(latent.shape)
            latent_codes.append(latent)
            test_name = batch['model_name']
            test_names = test_names + test_name
        latent_codes = torch.cat(latent_codes, dim=0)
        return latent_codes, test_names