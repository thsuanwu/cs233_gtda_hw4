"""
Part-Aware PC-AE.

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

class PartAwarePointcloudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, part_classifier, part_lambda=0.005):
        """ Part-aware AE initialization
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        :param part_classifier: nn.Module acting as the second decoding branch that classifies the point part
        labels.
        """
        super(PartAwarePointcloudAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.part_classifier = part_classifier
        
        self.part_lambda = part_lambda
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, pointclouds):
        # Takes pointclouds as input
        """Forward pass of the AE
            :param pointclouds: B x N x 3
        """
        x = torch.transpose(pointclouds, 1, 2)
        x = self.encoder(x)
        x_0 = torch.transpose(x, 1, 2)
        
        # 1. Decoder, as beofre
        x = self.decoder(x_0)
        
        # 2. Classification
        #print("squeezed: ", x_0.repeat(1, 1024, 1).shape)
        y = torch.cat([pointclouds, x_0.repeat(1, 1024, 1)], dim=2) 
        #print("concatenated: ", y.shape)
        y = torch.transpose(y, 1, 2) 
        #print("transposed: ", y.shape)
        y = self.part_classifier(y)
        #print("classifier: ", y.shape)
        return x,y
        

    def train_for_one_epoch(self, loader, optimizer, device='cuda'):
        """ Train the autoencoder for one epoch based on the Chamfer loss.
        :param loader: (train) pointcloud_dataset loader
        :param optimizer: torch.optimizer
        :param device: cuda? cpu?
        :return: (float), average loss for the epoch.
        """        
        self.train()
        joint_loss_meter = AverageMeter()
        chamfer_loss_meter = AverageMeter()
        xe_loss_meter = AverageMeter()
        
        ### My Work ###
        
        for batch in loader:
            optimizer.zero_grad()
            
            pointclouds = batch['point_cloud'].to(device)
            labels = batch['part_mask'].to(device)
            #print(labels.shape)
            x,y = self.forward(pointclouds)
            joint_loss = chamfer_loss(x, pointclouds).mean() + (self.part_lambda * self.cross_entropy_loss(y, labels).mean())
            joint_loss_meter.update(joint_loss, len(loader))
            chamfer_loss_meter.update(chamfer_loss(x, pointclouds).mean(), len(loader))
            xe_loss_meter.update(self.cross_entropy_loss(y, labels).mean(), len(loader))
            
            joint_loss.backward()
            optimizer.step()
        return joint_loss_meter.avg, chamfer_loss_meter.avg, xe_loss_meter.avg
    
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
        recon_losses = []
        pred_labels = []
        for batch in loader:
            pointclouds = batch["point_cloud"].to(device)
            labels = batch["part_mask"].to(device)
            #print(labels)
            x,y = self.forward(pointclouds)
            #print(y)
            joint_loss = chamfer_loss(x, pointclouds).mean() + (self.part_lambda * self.cross_entropy_loss(y, labels).mean())
            recon_loss = chamfer_loss(x, pointclouds).mean()
            
            losses.append(joint_loss)
            recon_losses.append(recon_loss)
            recons.append(x)
            pred_labels.append(y)
        return recons, losses, recon_losses, pred_labels
    
    def extract_latent_code(self, loader, device='cuda'):
        latent_codes = []
        test_names = []
        for batch in loader:
            pointclouds = batch["point_cloud"].to(device)
            x = torch.transpose(pointclouds, 1, 2)
            latent = self.embed(x)
            latent = latent.squeeze(-1)
            #print(latent.shape)
            latent_codes.append(latent)
            test_name = batch['model_name']
            test_names = test_names + test_name
        latent_codes = torch.cat(latent_codes, dim=0)
        return latent_codes, test_names