# Install the SAME prebuilt wheel into each deployment's existing runtime.
# CI supplies the upstream vLLM image tag through BASE_IMAGE.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG SPOOLCACHE_WHEEL
ARG SPOOLCACHE_WHEEL_SHA256
ARG SPOOLCACHE_COMMIT
COPY ${SPOOLCACHE_WHEEL} /opt/spoolcache-release/
RUN test -n "$SPOOLCACHE_WHEEL_SHA256" && test -n "$SPOOLCACHE_COMMIT" \
    && cd /opt/spoolcache-release \
    && test "$(find . -name '*.whl' | wc -l)" -eq 1 \
    && printf '%s  %s\n' "$SPOOLCACHE_WHEEL_SHA256" *.whl | sha256sum -c - \
    && python3 -m pip install --no-index --no-cache-dir --no-deps --force-reinstall ./*.whl
LABEL io.spoolcache.wheel.sha256=$SPOOLCACHE_WHEEL_SHA256 \
      io.spoolcache.commit=$SPOOLCACHE_COMMIT \
      org.opencontainers.image.source="https://github.com/xudongcc/spoolcache" \
      org.opencontainers.image.title="vLLM OpenAI with SpoolCache"
