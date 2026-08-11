# Parte de la imagen de vLLM a propósito: ya trae un PyTorch que funciona en
# este hardware (Blackwell / sm_121, ARM64). Instalar torch desde cero acá es
# la principal fuente de dolor en esta máquina.
FROM vllm/vllm-openai:latest

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Congela el torch/torchaudio que ya trae la imagen. Sin esto, pip es libre de
# reemplazarlos por wheels genéricos sin soporte para sm_121 y el contenedor
# arranca pero no ve la GPU.
RUN python3 -c "\
import torch, torchaudio; \
open('/tmp/constraints.txt','w').write(\
'torch==%s\ntorchaudio==%s\n' % (\
torch.__version__.split('+')[0], torchaudio.__version__.split('+')[0]))" \
    && cat /tmp/constraints.txt

# Sin pin de versión: pyannote 3.3.x usa torchaudio.AudioMetaData, que las
# versiones nuevas de torchaudio ya no exponen.
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
