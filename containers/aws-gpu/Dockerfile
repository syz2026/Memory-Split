FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training@sha256:1414a836532f22b271c03b7ccdbdff3daa0591975b3bd9a3cf51601a45b37f4f

LABEL org.opencontainers.image.base.name="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training@sha256:1414a836532f22b271c03b7ccdbdff3daa0591975b3bd9a3cf51601a45b37f4f" \
      org.opencontainers.image.platform="linux/amd64" \
      org.opencontainers.image.title="MemorySplit AWS GPU runtime"

ENV HOME=/home/memorysplit \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1

USER root

RUN groupadd --gid 10001 memorysplit \
    && useradd --create-home --gid 10001 --no-log-init \
        --shell /usr/sbin/nologin --uid 10001 memorysplit

COPY containers/aws-gpu/requirements.lock /opt/memorysplit/requirements.lock
COPY containers/aws-gpu/inspect_container.py /opt/memorysplit/inspect_container.py
COPY cluster/aws/qualification_worker.py /opt/memorysplit/cluster/aws/qualification_worker.py

RUN /opt/conda/bin/python -m pip install \
        --disable-pip-version-check \
        --force-reinstall \
        --no-cache-dir \
        --no-compile \
        --no-deps \
        --only-binary=:all: \
        --report=/opt/memorysplit/project-install-report.json \
        --require-hashes \
        --requirement /opt/memorysplit/requirements.lock \
    && /opt/conda/bin/python -c "import torch; assert torch.__version__.split('+', 1)[0] == '2.9.0'"

WORKDIR /workspace
USER 10001:10001
