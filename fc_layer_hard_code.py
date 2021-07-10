# Simple implementation

import time
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from datasets.sg2 import StyleGAN2_Data


class FC_Model(nn.Module):
    def __init__(self, z_dim=512, c_dim=136):
        super(FC_Model, self).__init__()

        # Block 1
        self.affine = spectral_norm(nn.Linear(c_dim, z_dim))
        self.post = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, z_dim)),
            nn.LeakyReLU(0.2)
        )

        # Block 2
        self.affine2 = spectral_norm(nn.Linear(c_dim, z_dim))
        self.post2 = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, z_dim)),
            nn.LeakyReLU(0.2)
        )

        # Block 3
        self.affine3 = spectral_norm(nn.Linear(c_dim, z_dim))
        self.post3 = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, z_dim)),
            nn.LeakyReLU(0.2)
        )

        # Block 4
        self.affine4 = spectral_norm(nn.Linear(c_dim, z_dim))
        self.post4 = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, z_dim)),
            nn.LeakyReLU(0.2)
        )

        # Block 5
        self.affine5 = spectral_norm(nn.Linear(c_dim, z_dim))
        self.post5 = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, z_dim)),
            nn.LeakyReLU(0.2)
        )

        # Block 6
        self.affine6 = spectral_norm(nn.Linear(c_dim, z_dim))
        self.post6 = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, z_dim)),
            # nn.LeakyReLU(0.2)
        )

    def forward(self, z, c):

        # # Block 1
        c_out = self.affine(c)
        z_out = self.adain(z, c_out)
        z_out = self.post(z_out)

        # Block 2
        c_out = self.affine2(c)
        z_out = self.adain(z_out, c_out)
        z_out = self.post2(z_out)

        # Block 3
        c_out = self.affine3(c)
        z_out = self.adain(z_out, c_out)
        z_out = self.post3(z_out)

        # Block 4
        c_out = self.affine4(c)
        z_out = self.adain(z_out, c_out)
        z_out = self.post4(z_out)

        # Block 5
        c_out = self.affine5(c)
        z_out = self.adain(z_out, c_out)
        z_out = self.post5(z_out)

        # Block 6
        c_out = self.affine6(c)
        z_out = self.adain(z_out, c_out)
        z_out = self.post6(z_out)

        return z_out

    def adain(self, x, y, eps=1e-5):
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        std_x = torch.std(x)
        std_y = torch.std(y)
        return (x - mean_x) / (std_x + eps) * std_y + mean_y


if __name__ == "__main__":

    lr = 0.0002
    z_dim = 512
    c_dim = 68 * 2
    num_epochs = 25
    batch_size = 512
    root = '../generated'

    print("Loading Datasets...")

    data = {
        'train': StyleGAN2_Data(root=root, split='train'),
        'val': StyleGAN2_Data(root=root, split='val')
        }

    data_loader = {
        'train': torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False),
        'val': torch.utils.data.DataLoader(data['val'], batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
        }

    print("Loading Complete!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FC_Model(z_dim=z_dim, c_dim=c_dim)

    print(model)

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    since = time.time()

    val_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):

        print('-' * 10)
        print(f'Epoch {epoch}/{num_epochs - 1}')

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
                    outputs = outputs

                    loss = criterion(outputs, latents)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                running_loss += loss.item() * latents.size(0)

            epoch_loss = running_loss / len(data_loader[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val':
                val_loss_history.append(epoch_loss)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # torch.save(best_model_wts, 'ckpt/model.pth')

        scheduler.step()

    time_elapsed = time.time() - since

    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:4f}')