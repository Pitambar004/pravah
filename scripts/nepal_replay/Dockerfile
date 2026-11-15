# Build on a verified linux/arm64 CUDA/PyTorch base. No raw data in this image.
# Team 12 v4.3.1 server validation and an actual CUDA job are still required.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/pravah
RUN python -c "import torch; assert torch.version.cuda, 'CUDA PyTorch base required'"
COPY scripts/nepal_replay/requirements-service.txt /opt/pravah/requirements-service.txt
RUN python -m pip install --no-cache-dir -r /opt/pravah/requirements-service.txt
COPY scripts/run_nepal_bundle.py /opt/pravah/run_nepal_bundle.py
ENV PYTHONUNBUFFERED=1 PRAVAH_ENABLE_TEST_FIXTURES=0
EXPOSE 8000 8501
