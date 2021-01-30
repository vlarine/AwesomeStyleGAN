CUDA_VISIBLE_DEVICES=0 python3 generate_video.py \
    --network=models/nvidia-stylegan2-ada-ffhq.pkl \
    --outdir=frames \
    --seeds=42-45 \
    --steps=100 \
    --fps=25 \
    --out=video.mp4


