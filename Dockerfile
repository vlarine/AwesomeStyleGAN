# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

ARG BASE_IMAGE=nvcr.io/nvidia/tensorflow:20.10-tf1-py3
FROM $BASE_IMAGE

RUN apt-get update && \
    apt-get install  -y tmux ffmpeg

RUN pip install --upgrade pip
COPY . /workdir/
RUN cd /workdir && pip install --no-cache-dir -r requirements.txt

RUN cd /workdir && git clone https://github.com/NVlabs/stylegan2-ada.git
RUN cd /workdir && wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl -O models/nvidia-stylegan2-ada-ffhq.pkl
