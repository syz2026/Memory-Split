# image.lock.json is the reviewed source of this digest.  The build planner
# always passes BASE_DIGEST explicitly and rejects an unpinned value.
ARG BASE_IMAGE=public.ecr.aws/deep-learning-containers/pytorch:2.12.1-cu130-amzn2023
ARG BASE_DIGEST=sha256:3bfc0b4c9561561b4cd578cb78611ec6258e1791854000b58a279d5880afc7d6
FROM ${BASE_IMAGE}@${BASE_DIGEST}

ARG BASE_DIGEST
ARG RUNTIME_UID=10001
ARG RUNTIME_GID=10001

USER root
COPY image.lock.json requirements.lock /opt/memorysplit-build/
RUN test "${BASE_DIGEST#sha256:}" != "${BASE_DIGEST}" \
    && test "${#BASE_DIGEST}" -eq 71 \
    && test -x /opt/venv/bin/python \
    && /opt/venv/bin/python -c \
        'import sys; assert sys.version_info[:2] == (3, 11)' \
    && /opt/venv/bin/python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --only-binary=:all: \
        --require-hashes \
        --requirement /opt/memorysplit-build/requirements.lock \
    && groupadd --gid "${RUNTIME_GID}" memorysplit \
    && useradd --create-home --uid "${RUNTIME_UID}" \
        --gid "${RUNTIME_GID}" --shell /sbin/nologin memorysplit \
    && install -d -m 0755 -o "${RUNTIME_UID}" -g "${RUNTIME_GID}" \
        /opt/memorysplit /workspace \
    && install -m 0444 -o "${RUNTIME_UID}" -g "${RUNTIME_GID}" \
        /opt/memorysplit-build/image.lock.json \
        /opt/memorysplit/image.lock.json \
    && install -m 0444 -o "${RUNTIME_UID}" -g "${RUNTIME_GID}" \
        /opt/memorysplit-build/requirements.lock \
        /opt/memorysplit/requirements.lock \
    && rm -rf /opt/memorysplit-build

# The two locks are build metadata, not application source. Source, datasets,
# checkpoints, and the external sealed evaluator are mounted at runtime.

ENV HOME=/home/memorysplit
WORKDIR /workspace
USER 10001:10001
ENTRYPOINT []
CMD ["/opt/venv/bin/python"]
