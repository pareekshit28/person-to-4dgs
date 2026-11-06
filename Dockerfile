# Optional reproducible base. Cloning into an existing RunPod devel pod also works.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu
WORKDIR /opt/person-to-4dgs
COPY . .
ENV PERSON4D_DATA_DIR=/workspace/person4d
EXPOSE 8000
# Setup requires a GPU for extension compilation and smoke checks, so run it on
# the pod once, after attaching persistent storage. It is intentionally not RUN.
CMD ["bash", "-lc", "bash scripts/setup_runpod.sh && bash scripts/start.sh"]
