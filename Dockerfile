FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

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
RUN mkdir -p /models \
    && curl -fL --retry 3 -o /models/BS-Roformer-SW.ckpt \
        https://github.com/Alontsoran/product-separator/releases/download/models/BS-Roformer-SW.ckpt \
    && curl -fL --retry 3 -o /models/BS-Roformer-SW.yaml \
        https://github.com/Alontsoran/product-separator/releases/download/models/BS-Roformer-SW.yaml \
    && curl -fL --retry 3 -o /models/MDX23C-DrumSep-aufr33-jarredou.ckpt \
        https://github.com/Alontsoran/product-separator/releases/download/models/MDX23C-DrumSep-aufr33-jarredou.ckpt \
    && curl -fL --retry 3 -o /models/config_drumsep_mdx23c.yaml \
        https://github.com/Alontsoran/product-separator/releases/download/models/config_drumsep_mdx23c.yaml

CMD ["python", "-u", "handler.py"]
