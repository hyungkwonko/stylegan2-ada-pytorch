import os
import pickle
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sefa")
    parser.add_argument("--feature", default='all', choices=['low', 'middle', 'high', 'all'])
    parser.add_argument("--mtype", default='xflip', choices=['default', 'mixing', 'xflip'])

    args = parser.parse_args()

    if args.mtype == 'default':
        ckpt = os.path.join('pretrained_models', 'network-snapshot-008467.pkl')
    elif args.mtype == 'mixing':
        ckpt = os.path.join('pretrained_models', 'network-snapshot-mixing-010080.pkl')
    elif args.mtype == 'xflip':
        ckpt = os.path.join('pretrained_models', 'network-snapshot-xflip-020643.pkl')
    else:
        raise AssertionError('check pretrained model type')

    with open(ckpt, 'rb') as f:
        obj = f.read()
    model = pickle.loads(obj, encoding='latin1')

    weights = []
    names = []

    if args.feature == 'low':
        layers = ['b4', 'b8', ]
    elif args.feature == 'middle':
        layers = ['b16', 'b32', 'b64']
    elif args.feature == 'high':
        layers = ['b128', 'b256']
    else:
        layers = ['']

    for name in model["G_ema"].state_dict():
        if ("synthesis" in name) and ("torgb" not in name) and ("affine.weight" in name):
            for layer in layers:
                if layer in name:
                    weight = model["G_ema"].state_dict()[name]
                    weights.append(weight)
                    names.append(name)
                    print(f"[INFO] append {len(names)}: {name}... {weight.shape}")
                    break

    W = torch.cat(weights, 0)
    print(f"[INFO] concatenated W.shape: {W.shape}")

    save_name = f'{args.mtype}-{args.feature}.pt'
    out_file = os.path.join('pretrained_models', save_name)
    eigvec = torch.svd(W).V.to("cpu")
    torch.save({"filename": save_name, "eigvec": eigvec}, out_file)
    print(f"[INFO] saved successfully to {out_file}")
