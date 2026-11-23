FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY separate_full.py handler.py ./

# The base image already has CUDA torch. Installing the separator must not
# replace it with a CPU build.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "audio-separator==0.47.0" runpod \
    && python -c "import torch; print('cuda', torch.cuda.is_available())"

ENV AUDIO_SEPARATOR_MODEL_DIR=/models
COPY BS-Roformer-SW.yaml config_drumsep_mdx23c.yaml /models/
# The old GitHub release that hosted these weights is gone. These are the
# public copies of the same two checkpoints. RunPod downloads them while it
# builds the image, so this Mac does not have to upload a gigabyte.
RUN curl -fL --retry 3 -o /models/BS-Roformer-SW.ckpt \
        https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt \
    && curl -fL --retry 3 -o /models/MDX23C-DrumSep-aufr33-jarredou.ckpt \
        https://huggingface.co/lainlives/audio-separator-models/resolve/main/MDX23C-DrumSep-aufr33-jarredou.ckpt

CMD ["python", "-u", "handler.py"]
