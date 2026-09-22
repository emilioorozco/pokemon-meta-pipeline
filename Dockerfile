# The event consumer as a container: one long-running process on one queue.
# Every other stage is a batch command; this is the only piece that waits.
FROM python:3.12-slim

# uv installs from the committed lock file, so the image runs the versions CI
# tested rather than whatever resolves on build day.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependencies first, without the project itself: this layer is rebuilt only
# when the lock file changes, not on every edit to the pipeline.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the code, and the project on top of the cached dependency layer.
COPY pipeline ./pipeline
RUN uv sync --frozen --no-dev

# The lake is a mounted volume (see compose.yaml), so nothing the consumer
# writes is kept in the image.
ENV PIPELINE_DATA_DIR=/app/data
# The environment's interpreter, so uv is not needed at run time.
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "pipeline.consume"]
