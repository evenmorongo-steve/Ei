# =====================================================================================
# Ei — containerised analysis environment.
#
#   docker build -t ei:0.1.0 .
#   docker run --rm -v "$PWD/runs:/work/runs" ei:0.1.0 run configs/experiment.yaml --fast
#
# Design notes:
#   * CPU base by default. The analysis (rSVD, eigendecomposition, surrogates) is
#     BLAS-bound, not GPU-bound. GPUs are only needed for the ACTIVATION EXTRACTION stage,
#     which is pluggable -- see the gpu stage below.
#   * Multi-stage so the runtime image carries no compiler toolchain.
#   * Non-root user: a container that writes root-owned files into a mounted results
#     volume is a daily annoyance and an audit problem.
#   * Threads pinned: BLAS thread count changes floating-point reduction order and hence
#     the last digits of eigenvalues.
# =====================================================================================

# ------------------------------------------------------------------ builder
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran libopenblas-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel && pip install .

# ------------------------------------------------------------------ runtime
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="Ei" \
      org.opencontainers.image.description="Eigenmode instrumentation for persistent-structure interpretability" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/evenmorongo-steve/Ei" \
      ei.claim-ceiling="computational correlate only; no phenomenality claims"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 ei

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

WORKDIR /work
COPY --chown=ei:ei configs ./configs
COPY --chown=ei:ei scripts ./scripts
COPY --chown=ei:ei tests ./tests
COPY --chown=ei:ei docs ./docs
RUN mkdir -p /work/runs && chown -R ei:ei /work
USER ei

# Fails the build if the instrument cannot recover a planted mode from synthetic ground
# truth. A container that ships a miscalibrated instrument is worse than no container.
RUN python -c "import ei, ei.modes, ei.pipeline; print('ei', ei.__version__, 'import OK')" \
 && python -m ei.cli selftest

HEALTHCHECK --interval=60s --timeout=20s --retries=2 \
    CMD python -c "import ei; import sys; sys.exit(0)"

ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "ei.cli"]
CMD ["--help"]

# ------------------------------------------------------------------ gpu (optional)
# Only needed for the pluggable activation-extraction backend.
#   docker build --target gpu -t ei:0.1.0-gpu .
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 ei

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=8

WORKDIR /work
COPY --chown=ei:ei configs ./configs
COPY --chown=ei:ei scripts ./scripts
RUN mkdir -p /work/runs && chown -R ei:ei /work
USER ei

# Torch is installed here rather than in the builder so the CUDA build matches the base
# image's driver ABI.
RUN pip install --no-cache-dir "torch>=2.1" "transformer-lens>=2.0" "transformers>=4.40"

ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "ei.cli"]
CMD ["--help"]
