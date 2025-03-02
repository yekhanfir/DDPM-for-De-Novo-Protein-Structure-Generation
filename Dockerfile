FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml /app/environment.yml

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "protein-diffusion", "/bin/bash", "-c"]


# Add "src/" to the PYTHONPATH
# to allow import from src/__init__.py
ENV PYTHONPATH=/app/src:$PYTHONPATH