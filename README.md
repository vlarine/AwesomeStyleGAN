# AwesomeStyleGAN
Awesome StyleGAN models

## GAN image transformation video

Generate [video](media/video.mp4) of image transformation using offficial Nvidia StyleGAN2 ADA models:

* [TensorFlow](https://github.com/NVlabs/stylegan2-ada) implementation.
* [Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) implementation.

![video.gif](media/video.gif)

### Installation

Use [Docker](Dockerfile)

Or manually install all dependencies and download models:

#### TensorFlow model

Clone the repository:

```
git clone https://github.com/NVlabs/stylegan2-ada.git
```

Download the preptarained model:

```
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl -O models/nvidia-stylegan2-ada-ffhq-tf.pkl
```

#### Pytorch model

Clone the repository:

```
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```

Download the preptarained model:

```
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -O models/nvidia-stylegan2-ada-ffhq-pt.pkl
```

### Usage

An example script usage is in the [scripts folder](scripts):

* [TensorFlow script](scripts/run_generate_video.sh)
* [Pytorch script](scripts/run_generate_video_pytorch.sh)


