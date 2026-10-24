# Hosted build: Example and Upload only (HOSTED=1 turns the live camera off).
# Build:  docker build -t pose-analyzer .
# Run:    docker run -p 7860:7860 pose-analyzer      then open http://localhost:7860
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Hosting platforms run containers as a non-root user.
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/pose-analyzer

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn==23.0.0

COPY --chown=app:app . .
RUN mkdir -p models runs && chown app:app models runs
USER app

# The pose models are not in the repository.
RUN for m in full heavy; do \
      curl -fL -o models/pose_landmarker_$m.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_$m/float16/latest/pose_landmarker_$m.task; \
    done

ENV HOSTED=1 PORT=7860
EXPOSE 7860
# One worker: upload jobs are tracked in memory. Threads handle several visitors at once.
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 300 "app:create_app()"
