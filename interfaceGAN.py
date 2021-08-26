import os
import torch
import argparse
import dnnlib
import numpy as np
from PIL import Image
import legacy
from torchvision import utils
from datetime import datetime
from sklearn import svm
import torchvision.transforms as transforms



MANIPULATION_TARGET = 'gender'
# MANIPULATION_TARGET = 'pose'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="interfaceGAN")
    parser.add_argument("--ckpt", default='pretrained_models/network-snapshot-xflip-020643.pkl')
    parser.add_argument('--truncation_psi', type=float, help='Truncation psi', default=0.8)
    parser.add_argument('--noise-mode', choices=['const', 'random', 'none'], default='const')
    parser.add_argument('--outdir', type=str, default='outdir')
    parser.add_argument('--space', choices=['z', 'w'], default='w')
    parser.add_argument('--part1', type=str, default=f'pos_{MANIPULATION_TARGET}_indices.npy')
    parser.add_argument('--part2', type=str, default=f'neg_{MANIPULATION_TARGET}_indices.npy')
    parser.add_argument("--svm_train_iter", type=int, default=10000)
    parser.add_argument("--save_file", type=str, default=f"manipulation/{MANIPULATION_TARGET}.pt")
    parser.add_argument("--num_imgs", type=int, default=15)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device('cuda')
    with dnnlib.util.open_url(args.ckpt) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    weights = []
    names = []

    part1_indexes = np.load(os.path.join('manipulation', args.part1)).astype(int)
    part2_indexes = np.load(os.path.join('manipulation', args.part2)).astype(int)

    # get boundary using two parts.
    testset_ratio = 0.1
    np.random.shuffle(part1_indexes)
    np.random.shuffle(part2_indexes)

    positive_len = len(part1_indexes)
    negative_len = len(part2_indexes)

    pos_latents = []
    neg_latents = []

    label = torch.zeros([1, G.c_dim], device=device)
    for seed in part1_indexes:
        latent = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        if args.space == 'w':
            latent = G.mapping(latent, label)
        latent = latent.cpu().numpy().squeeze()

        if args.space == 'w':
            pos_latents.append(latent[0])
        else:
            pos_latents.append(latent)
    pos_latents = np.array(pos_latents)

    for seed in part2_indexes:
        latent = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device) 
        if args.space == 'w':
            latent = G.mapping(latent, label)
        latent = latent.cpu().numpy().squeeze()
        if args.space == 'w':
            neg_latents.append(latent[0])
        else:
            neg_latents.append(latent)
    neg_latents = np.array(neg_latents)

    positive_train = pos_latents[int(positive_len * testset_ratio) :]
    positive_val = pos_latents[: int(positive_len * testset_ratio)]

    negative_train = neg_latents[int(negative_len * testset_ratio) :]
    negative_val = neg_latents[: int(negative_len * testset_ratio)]

    # Training set.
    train_data = np.concatenate([positive_train, negative_train], axis=0)
    train_label = np.concatenate(
        [np.ones(len(positive_train), dtype=np.int), np.zeros(len(negative_train), dtype=np.int),],
        axis=0
    )

    # Validation set.
    val_data = np.concatenate([positive_val, negative_val], axis=0)
    val_label = np.concatenate(
        [np.ones(len(positive_val), dtype=np.int), np.zeros(len(negative_val), dtype=np.int),],
        axis=0,
    )

    print(
        f"positive_train: {len(positive_train)}, \
        negative_train:{len(negative_train)}, \
        positive_val:{len(positive_val)}, \
        negative_val:{len(negative_val)}"
    )

    print(f"Training boundary. {datetime.now()}")
    clf = svm.SVC(kernel="linear", max_iter=args.svm_train_iter, verbose=True)
    classifier = clf.fit(train_data, train_label)
    print(f"Finish training. {datetime.now()}")

    print(f"validate boundary.")
    val_prediction = classifier.predict(val_data)
    
    correct_num = np.sum(val_label == val_prediction)

    print(
        f"Accuracy for validation set: "
        f"{correct_num} / {len(val_data)} = "
        f"{correct_num / (len(val_data)):.6f}"
    )

    print("classifier.coef_.shape", classifier.coef_.shape)
    boundary = classifier.coef_.reshape(1, -1).astype(np.float32)
    boundary = boundary / np.linalg.norm(boundary)
    print("boundary.shape", boundary.shape)

    boundary = torch.from_numpy(boundary).float().to(device)

    torch.save(
        {
            "boundary": boundary,
            "part1_indexes": part1_indexes,
            "part2_indexes": part2_indexes,
        },
        args.save_file,
    )

    images = []
    seeds = np.random.choice(range(50000), args.num_imgs, replace=False)
    for seed in seeds:
        for i, distance in enumerate(range(-10, 11, 2)):
            # print("distance: ", distance)
            latent = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            if args.space == 'w':
                latent = G.mapping(latent, label, truncation_psi=args.truncation_psi)
                latent += distance * boundary
                img = G.synthesis(latent)
            else:
                latent += distance * boundary
                img = G(latent, label, truncation_psi=args.truncation_psi, noise_mode=args.noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # Image.fromarray(img[0].cpu().numpy(), 'RGB').resize((256, 256), Image.ANTIALIAS).save(f'{args.outdir}/seed{seed:05d}_{i}_{distance}.jpg')
            img = transforms.ToTensor()(Image.fromarray(img[0].cpu().numpy(), 'RGB').resize((256, 256), Image.ANTIALIAS))
            images.append(img)

    utils.save_image(
        images,
        f"{args.outdir}/{MANIPULATION_TARGET}_result.jpg",
        nrow=11,
        normalize=True,
        range=(-1, 1),
    )

    print("image saved successfully...")
