import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

from datasets.sg2 import StyleGAN2_Data
from fc_layer import FC_Model


def test(model, data_loader, criterion, device):

    model.eval()

    running_loss = 0.0

    for batch in tqdm(data_loader):

        latents = batch['latent'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(latents, labels)
            outputs = outputs[0][0]
            loss = criterion(outputs, latents)

        running_loss += loss.item() * latents.size(0)

    epoch_loss = running_loss / len(data_loader.dataset)

    print(f'Loss: {epoch_loss:.4f}')
    print(f"Sample latents {latents[0][:7].cpu()}")
    print(f"Sample output: {outputs[0][:7].cpu()}")



def main():

    parser = argparse.ArgumentParser("MLP layer (auxiliary network) test")

    parser.add_argument('--z_dim', type=int, default=512, help='latent_dim')
    parser.add_argument('--c_dim', type=int, default=136, help='class_dim')
    parser.add_argument('--num_mlp_layers', type=int, default=6, help='number of mlp layers')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--root', type=str, default='../generated', help='training data dir')
    parser.add_argument('--ckpt', type=str, default='ckpt/model_0.0002_8_6_1e-08.pth', help='model checkpoint save path')

    args = parser.parse_args()

    data = StyleGAN2_Data(root=args.root, split='test')
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FC_Model(z_dim=args.z_dim, c_dim=args.c_dim, n=args.num_mlp_layers)
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)

    criterion = nn.MSELoss()

    test(model, data_loader, criterion, device)



if __name__ == "__main__":
    main()