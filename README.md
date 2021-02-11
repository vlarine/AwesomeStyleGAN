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


### Video with sound 

Instead of linear interpolation you can use sine interpolation. There is an example of viddeo with beat detection from [Aubio](https://github.com/aubio/aubio) library.

* [Run script](scripts/run_generate_video_beats.sh)
* [Input audio file](media/beats.wav)
* [Result video file](media/beats.mp4)

There is an [another example](https://www.tiktok.com/@vlarine/video/6927257593178033413) of video with beat detection in the wild.



