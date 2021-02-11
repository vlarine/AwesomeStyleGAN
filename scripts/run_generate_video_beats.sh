CUDA_VISIBLE_DEVICES=0 python3 generate_video_beats.py \
    --network=models/nvidia-stylegan2-ada-ffhq-pt.pkl \
    --outdir=frames \
    --seed=42 \
    --fps=25 \
    --wav=media/beats.wav \
    --out=beats.mp4


