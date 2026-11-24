FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 libgomp1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY separate_full.py handler.py ./

# The base image already has CUDA torch. Installing the separator must not
# replace it with a CPU build.
# basic-pitch 0.4 on Linux Python 3.11 depends on TensorFlow, and that install
# failed the image. The note model in the package is ONNX. Install it without
# TensorFlow.
# Whisper is not baked into the image. Hugging Face downloads fail on this
# builder. The first lyrics job pulls the Hebrew model onto the card.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "audio-separator==0.47.0" runpod faster-whisper "av==14.2.0"

RUN pip install --no-cache-dir --no-deps "basic-pitch==0.4.0" \
    && pip install --no-cache-dir "onnxruntime==1.20.1" "pretty-midi>=0.2.9" "mir_eval>=0.6" \
    && python -c "import torch; print('cuda', torch.cuda.is_available())" \
    && python -c "from basic_pitch import FilenameSuffix, build_icassp_2022_model_path; p=build_icassp_2022_model_path(FilenameSuffix.onnx); assert p.is_file(), p; print('notes', p.name)"

ENV AUDIO_SEPARATOR_MODEL_DIR=/models
ENV WHISPER_MODEL=/models/whisper-he
COPY BS-Roformer-SW.yaml config_drumsep_mdx23c.yaml /models/
# The old GitHub release that hosted these weights is gone. These are the
# public copies of the same two checkpoints. RunPod downloads them while it
# builds the image, so this Mac does not have to upload a gigabyte.
RUN curl -fL --retry 5 --retry-delay 10 -o /models/BS-Roformer-SW.ckpt \
        https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt \
    && curl -fL --retry 5 --retry-delay 10 -o /models/MDX23C-DrumSep-aufr33-jarredou.ckpt \
        https://huggingface.co/lainlives/audio-separator-models/resolve/main/MDX23C-DrumSep-aufr33-jarredou.ckpt

CMD ["python", "-u", "handler.py"]
