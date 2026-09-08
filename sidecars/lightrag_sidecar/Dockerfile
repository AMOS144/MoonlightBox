FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /service
COPY pyproject.toml README.md ./
COPY src ./src
# Keep this image buildable with both BuildKit and Docker's legacy builder.
# The cache mount is only an optimization and is unavailable in the legacy
# builder used by the current WSL Docker installation.
RUN pip install .

RUN useradd --create-home --uid 10001 sidecar \
    && mkdir -p /var/lib/moonlightbox-lightrag \
    && chown -R sidecar:sidecar /var/lib/moonlightbox-lightrag
USER sidecar

EXPOSE 9621
CMD ["moonlightbox-lightrag-sidecar"]
