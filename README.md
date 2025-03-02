# DDPM-for-De-Novo-Protein-Sequence-Generation
In this repo, we implement and train a Denoising Diffusion Probabilistic Model (DDPM) to generate De Novo functional Proteins.

## Build image
docker build -t protein-diffusion .

## Run container
docker run --rm --gpus all --volume src:/app -d -it protein-diffusion

## Attach container
docker container attach "container_ID"