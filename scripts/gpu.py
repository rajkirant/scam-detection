#!/usr/bin/env python3
"""
gpu.py - one answer to "which device", for every part of the project that
can use one.

The GPU when there is one, the CPU when there is not. SCAM_DEVICE=cpu forces
the CPU for everything that asks here - worth it only on a machine where
something else needs the VRAM.

Ollama is not covered: it picks its own device and always uses the GPU when
the model fits. BERT training and the BERT benchmark have their own --cpu.
What this governs is the smaller models that would otherwise sit on the CPU
by default - the sentence-transformer that embeds queries for the retrieval
systems, and the BERT page's scoring.
"""

import os


def device():
    forced = os.environ.get("SCAM_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        # no torch (the system python the web UI runs on) means no CUDA to use
        return "cpu"
