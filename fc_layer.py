# reference
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
# https://gist.github.com/rosinality/a96c559d84ef2b138e486acf27b5a56e

import os
import time
import copy
import logging

import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from datasets.sg2 import StyleGAN2_Data


# multiple inputs reference: https://github.com/pytorch/pytorch/issues/19808#
class MultiSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class FC_Block(nn.Module):
    def __init__(self, z_dim=512, c_dim=136):
        super(FC_Block, self).__init__()

        self.affine = spectral_norm(nn.Linear(c_dim, z_dim))
        self.post = MultiSequential(
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(z_dim, z_dim)),
        )

    def forward(self, z_in, c):
        c_out = self.affine(c)
        z_out = self.adain(z_in, c_out)
        z_out = self.post(z_out)
        return z_out, c

    def adain(self, x, y, eps=1e-5):
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        std_x = torch.std(x)
        std_y = torch.std(y)
        return (x - mean_x) / (std_x + eps) * std_y + mean_y


class FC_Model(nn.Module):
    def __init__(self, z_dim=512, c_dim=136, n=6):
        super(FC_Model, self).__init__()

        self.model = MultiSequential(
            *self._make_layer(FC_Block, z_dim, c_dim, n)
        )

    def _make_layer(self, block, z_dim, c_dim, n):
        layers = []
        for _ in range(n):
            layers.append(block(z_dim, c_dim))
        return layers

    def forward(self, z, c):
        z_out = self.model(z, c)
        return z_out, c


def train(args, model, data_loader, criterion, optimizer, scheduler, device):

    since = time.time()

    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(args.num_epochs):

        logging.info('-' * 10)
        logging.info(f'Epoch {epoch}/{args.num_epochs - 1}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            # Iterate over data.
            for batch in tqdm(data_loader[phase]):

                latents = batch['latent'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(latents, labels)
                    outputs = outputs[0][0]

                    loss = criterion(outputs, latents)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                running_loss += loss.item() * latents.size(0)

            epoch_loss = running_loss / len(data_loader[phase].dataset)

            logging.info(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val':
                val_loss_history.append(epoch_loss)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(args.ckpt_dir, f'model_{args.lr}_{args.batch_size}_{args.num_mlp_layers}.pth'))

        scheduler.step()

    time_elapsed = time.time() - since

    logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logging.info(f'Best val Loss: {best_loss:4f}')

    model.load_state_dict(best_model_wts)

    return model, val_loss_history


def main():

    parser = argparse.ArgumentParser("PyTorch SE-Net Fine-tuning Code")

    parser.add_argument('--z_dim', type=int, default=512, help='latent_dim')
    parser.add_argument('--c_dim', type=int, default=136, help='class_dim')
    parser.add_argument('--num_mlp_layers', type=int, default=6, help='number of mlp layers')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=40, help='number of epochs to run')

    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 norm')

    parser.add_argument('--root', type=str, default='../generated', help='training data dir')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt', help='model checkpoint save path')
    parser.add_argument('--log_dir', type=str, default='log', help='save directory for log file')

    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, f'senet_{datetime.now().strftime("%H:%M:%S")}.log')),
            logging.StreamHandler()
        ]
    )

    logging.info("Loading Datasets...")

    data = {
        'train': StyleGAN2_Data(root=args.root, split='train'),
        'val': StyleGAN2_Data(root=args.root, split='val')
        }

    data_loader = {
        'train': torch.utils.data.DataLoader(data['train'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False),
        'val': torch.utils.data.DataLoader(data['val'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
        }

    logging.info("Loading Complete!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FC_Model(z_dim=args.z_dim, c_dim=args.c_dim, n=args.num_mlp_layers)

    logging.info(model)

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    model, val_history = train(args, model, data_loader, criterion, optimizer, scheduler, device)

    logging.info(val_history)
    logging.info('Save validation history')
    logging.info('-' * 10)
    logging.info('Successfully finished training!')



if __name__ == "__main__":
    main()