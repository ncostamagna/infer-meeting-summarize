# Based on the vLLM image on purpose: it already ships a PyTorch that works on
# this hardware (Blackwell / sm_121, ARM64). Installing torch from scratch here
# is the main source of pain on this machine.
FROM vllm/vllm-openai:latest

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Freeze the torch/torchaudio already present in the image. Without this, pip is
# free to replace them with generic wheels lacking sm_121 support, and the
# container then starts but cannot see the GPU.
RUN python3 -c "\
import torch, torchaudio; \
open('/tmp/constraints.txt','w').write(\
'torch==%s\ntorchaudio==%s\n' % (\
torch.__version__.split('+')[0], torchaudio.__version__.split('+')[0]))" \
    && cat /tmp/constraints.txt

# No version pin: pyannote 3.3.x uses torchaudio.AudioMetaData, which newer
# torchaudio releases no longer expose.
RUN python3 -m pip install --no-cache-dir -c /tmp/constraints.txt \
    pyannote.audio \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    python-multipart

RUN python3 -c "\
import torch, torchaudio, pyannote.audio; \
print('torch', torch.__version__, '| torchaudio', torchaudio.__version__, \
'| pyannote', pyannote.audio.__version__)"

WORKDIR /app
COPY app.py /app/app.py

ENTRYPOINT ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
