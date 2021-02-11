# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network file."""

import sys
import argparse
import os
import re

import numpy as np
import PIL.Image

sys.path.insert(0, 'stylegan2-ada-pytorch')
import dnnlib
import torch
import legacy

from tqdm import tqdm

from utils import get_beats


#----------------------------------------------------------------------------

def seed2vec(G, seed):
    rnd = np.random.RandomState(seed)
    return rnd.randn(1, G.z_dim)

#----------------------------------------------------------------------------

def generate_video(network_pkl, seed, fps, output_filename, wav_filename, outdir):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    label = torch.zeros([1, G.c_dim], device=device)

    os.makedirs(outdir, exist_ok=True)

    beats, total_len = get_beats(wav_filename)
    beats = [int(np.round(fps * x)) for x in beats[::1]]

    # Add the first and the last frames to beats array
    if 0 not in beats:
        beats = [0] + beats
    last = int(np.round(fps * total_len))
    if last not in beats:
        beats.append(last)

    idx = 0
    for i in tqdm(range(len(beats)-1)):
        v1 = seed2vec(G, seed)
        seed += 1
        v2 = seed2vec(G, seed)
        diff = v2 - v1

        n_frames = beats[i+1] - beats[i]

        x = np.linspace(0, np.pi, n_frames)
        y = np.cumsum(1 - np.sin(x))
        y /= y[-1]

        for j in range(n_frames):
            current = v1 + diff * y[j]
            z = torch.from_numpy(current).to(device)
            img = G(z, label, noise_mode='const')
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(os.path.join(outdir, 'frame-{}.png'.format(beats[i] + j)))

    cmd = 'ffmpeg -y -i {}/frame-%d.png -i {} -r {} -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p {}'.format(outdir, wav_filename, fps, output_filename)
    os.system(cmd)


#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate images using pretrained network file.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network', help='Network filename', dest='network_pkl', required=True)
    parser.add_argument('--seed', dest='seed', type=int, help='Random seed', default=42)
    parser.add_argument('--fps', dest='fps', type=int, help='Output video FPS', default=25)
    parser.add_argument('--out', dest='output_filename', help='Output video filename', required=True)
    parser.add_argument('--wav', dest='wav_filename', help='Input wav filename', required=True)
    parser.add_argument('--outdir', help='Where to save the output images', required=True, metavar='DIR')

    args = parser.parse_args()
    generate_video(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
