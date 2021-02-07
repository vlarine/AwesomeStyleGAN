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


#----------------------------------------------------------------------------

def seed2vec(G, seed):
    rnd = np.random.RandomState(seed)
    return rnd.randn(1, G.z_dim)

#----------------------------------------------------------------------------

def generate_video(network_pkl, seeds, steps, fps, output_filename, outdir):
    if len(seeds) < 2:
        print('Need more than one seed')
        return

    if steps < 1:
        print('At least one step required')
        return

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    label = torch.zeros([1, G.c_dim], device=device)

    os.makedirs(outdir, exist_ok=True)

    # Generate the images for the video.
    idx = 0
    for i in range(len(seeds)-1):
        v1 = seed2vec(G, seeds[i])
        v2 = seed2vec(G, seeds[i+1])

        diff = v2 - v1
        step = diff / steps
        current = v1.copy()

        for j in tqdm(range(steps), desc=f"Seed {seeds[i]}"):
            current = current + step
            z = torch.from_numpy(current).to(device)
            img = G(z, label, noise_mode='const')
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(os.path.join(outdir, f'frame-{idx}.png'))
            idx += 1

    cmd = 'ffmpeg -y -i {}/frame-%d.png -r {} -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p {}'.format(outdir, fps, output_filename)
    os.system(cmd)


#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate images using pretrained network file.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network', help='Network filename', dest='network_pkl', required=True)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--seeds', type=_parse_num_range, help='List of random seeds')
    parser.add_argument('--steps', dest='steps', type=int, help='Number of frames between the seeds', default=100)
    parser.add_argument('--fps', dest='fps', type=int, help='Output video FPS', default=25)
    parser.add_argument('--out', dest='output_filename', help='Output video filename', required=True)
    parser.add_argument('--outdir', help='Where to save the output images', required=True, metavar='DIR')

    args = parser.parse_args()
    generate_video(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
