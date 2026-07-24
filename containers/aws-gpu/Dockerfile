# image.lock.json is the reviewed source of this digest.  The build planner
# always passes BASE_DIGEST explicitly and rejects an unpinned value.
ARG BASE_IMAGE=public.ecr.aws/deep-learning-containers/pytorch:2.12.1-cu130-amzn2023
ARG BASE_DIGEST=sha256:3bfc0b4c9561561b4cd578cb78611ec6258e1791854000b58a279d5880afc7d6
FROM ${BASE_IMAGE}@${BASE_DIGEST}

ARG BASE_DIGEST
ARG RUNTIME_UID=10001
ARG RUNTIME_GID=10001

USER root
RUN test "${BASE_DIGEST#sha256:}" != "${BASE_DIGEST}" \
    && test "${#BASE_DIGEST}" -eq 71 \
    && groupadd --gid "${RUNTIME_GID}" memorysplit \
    && useradd --create-home --uid "${RUNTIME_UID}" \
        --gid "${RUNTIME_GID}" --shell /sbin/nologin memorysplit \
    && install -d -m 0755 -o "${RUNTIME_UID}" -g "${RUNTIME_GID}" \
        /opt/memorysplit /workspace

# The lock is build metadata, not application source.  Source and datasets are
# mounted read-only by the launcher at runtime.
COPY --chown=memorysplit:memorysplit image.lock.json /opt/memorysplit/image.lock.json

ENV HOME=/home/memorysplit
WORKDIR /workspace
USER 10001:10001
ENTRYPOINT []
CMD ["/opt/conda/bin/python"]
