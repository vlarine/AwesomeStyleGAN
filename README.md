# AwesomeStyleGAN
Awesome StyleGAN models

## GAN image transformation video

Generate [video](media/video.mp4) of image transformation using [Nvidia StyleGAN2 ADA](https://github.com/NVlabs/stylegan2-ada) model.

![video.gif](media/video.gif)

### Installation

Clone the repository:

```
git clone https://github.com/NVlabs/stylegan2-ada.git
```

Download the preptarained model:

```
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl -O models/nvidia-stylegan2-ada-ffhq.pkl
```

Or use [docker](Dockerfile).

### Usage

An example script usage is in the [scripts folder](scripts/run_generate_video.sh)


