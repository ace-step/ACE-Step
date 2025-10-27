#!/bin/bash
# ACE-Step startup script
# Runs on port 7866, accessible on local network and Tailscale
# Using PyTorch with CUDA 13.0 for RTX 5090 support

source .venv/bin/activate
acestep \
  --server_name 0.0.0.0 \
  --port 7866 \
  --bf16 true \
  --torch_compile false \
  --cpu_offload true \
  --overlapped_decode true
